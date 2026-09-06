using System.Text.RegularExpressions;
using Google.GenAI;
using Google.GenAI.Types;
using WinAiBuddy.Models;

namespace WinAiBuddy.Services;

public sealed class GeminiLiveSessionService : IAsyncDisposable
{
    private const int MaxReconnectAttempts = 5;
    private const int RapidCloseLimit = 3;
    private static readonly TimeSpan RapidCloseWindow = TimeSpan.FromSeconds(10);
    private static readonly TimeSpan InterruptDebounce = TimeSpan.FromMilliseconds(250);
    private const string InvalidArgumentMarker = "Request contains an invalid argument";

    private static readonly Regex DuplicateWhitespaceRegex = new(@"\s{2,}", RegexOptions.Compiled);
    private static readonly Regex PunctuationSpacingRegex = new(@"(?<=[,!?;:.])(?=\p{L})", RegexOptions.Compiled);

    private readonly SemaphoreSlim _sendLock = new(1, 1);
    private readonly SemaphoreSlim _reconnectLock = new(1, 1);
    private readonly object _transcriptLock = new();
    private readonly object _resumptionLock = new();
    private readonly DiagnosticsLogService _diagnosticsLogService;

    private Client? _client;
    private AsyncSession? _session;
    private CancellationTokenSource? _receiveLoopCts;
    private CancellationTokenSource? _serviceLifetimeCts;
    private Task? _receiveLoopTask;
    private AppSettings? _activeSettings;
    private string? _sessionResumptionHandle;

    private bool _shouldBeRunning;
    private bool _googleSearchEnabled = true;
    private int _recoveryInFlight;
    private DateTime _connectedAt = DateTime.MinValue;
    private int _rapidCloseCount;
    private DateTime _lastInterruptedAt = DateTime.MinValue;

    private string _currentInputTranscript = string.Empty;
    private string _currentOutputTranscript = string.Empty;
    private DateTime _lastInputTranscriptAt = DateTime.MinValue;
    private DateTime _lastOutputTranscriptAt = DateTime.MinValue;
    private bool _inputTurnOpen;
    private bool _outputTurnOpen;

    public event Action<string>? StatusChanged;
    public event Action<bool>? SessionStateChanged;
    public event Action<string>? InputTranscriptionChanged;
    public event Action<string>? OutputTranscriptionChanged;
    public event Action<byte[]>? AudioReceived;
    public event Action? Interrupted;

    public GeminiLiveSessionService(DiagnosticsLogService diagnosticsLogService)
    {
        _diagnosticsLogService = diagnosticsLogService;
    }

    public async Task StartAsync(
        AppSettings settings,
        IReadOnlyList<ConversationLogEntryRecord>? restoredConversation = null,
        CancellationToken cancellationToken = default)
    {
        await StopAsync(cancellationToken);
        ResetTranscriptState();
        ResetConnectionState();

        if (string.IsNullOrWhiteSpace(settings.ApiKey))
        {
            throw new InvalidOperationException("A Gemini API key is required.");
        }

        _activeSettings = settings;
        _shouldBeRunning = true;
        _serviceLifetimeCts = new CancellationTokenSource();

        var sessionLogPath = _diagnosticsLogService.StartSession(settings.LiveModel);
        LogSession(
            "Live",
            $"Start requested | model={settings.LiveModel} | voice={settings.Voice} | streamScreen={settings.StreamScreenFrames} | restoredTurns={restoredConversation?.Count ?? 0} | log={sessionLogPath}");

        if (IsGemini31Live(settings.LiveModel) && IsThinkingDisabled(settings))
        {
            LogSession(
                "Compat",
                "Gemini 3.1 Live uses thinkingLevel instead of thinkingBudget. Disabled is mapped to minimal for lowest latency.");
        }

        await ConnectAsync(settings, allowResumption: false, cancellationToken);

        if (restoredConversation is { Count: > 0 } restoredTurns)
        {
            await RestoreConversationContextAsync(restoredTurns, cancellationToken);
        }

        SessionStateChanged?.Invoke(true);
        PublishStatus("Connected to Gemini Live.");
    }

    public async Task SendAudioChunkAsync(AudioChunk chunk, CancellationToken cancellationToken = default)
    {
        var session = _session;
        if (!_shouldBeRunning || session is null)
        {
            return;
        }

        await SendRealtimeInputAsync(
            session,
            new LiveSendRealtimeInputParameters
            {
                Audio = new Blob
                {
                    Data = chunk.Bytes.ToArray(),
                    MimeType = chunk.MimeType
                }
            },
            "audio",
            cancellationToken);
    }

    public async Task SendVideoFrameAsync(ScreenshotCapture frame, CancellationToken cancellationToken = default)
    {
        var session = _session;
        if (!_shouldBeRunning || session is null)
        {
            return;
        }

        await SendRealtimeInputAsync(
            session,
            new LiveSendRealtimeInputParameters
            {
                Video = new Blob
                {
                    Data = frame.Bytes.ToArray(),
                    MimeType = frame.MimeType
                }
            },
            "video",
            cancellationToken);
    }

    public async Task StopAsync(CancellationToken cancellationToken = default)
    {
        _shouldBeRunning = false;
        _serviceLifetimeCts?.Cancel();

        await DisposeCurrentSessionAsync(resetTranscripts: true, cancellationToken);

        _serviceLifetimeCts?.Dispose();
        _serviceLifetimeCts = null;
        _activeSettings = null;
        ResetConnectionState();

        LogSession("Live", "Stop completed.");
        _diagnosticsLogService.EndSession("Stopped");
        SessionStateChanged?.Invoke(false);
    }

    public async ValueTask DisposeAsync()
    {
        await StopAsync();
        _sendLock.Dispose();
        _reconnectLock.Dispose();
    }

    private async Task ConnectAsync(AppSettings settings, bool allowResumption, CancellationToken cancellationToken)
    {
        var apiVersion = RequiresV1Alpha(settings) ? "v1alpha" : "v1beta";
        var resumptionHandle = allowResumption ? GetResumptionHandle() : null;

        LogSession(
            "Connect",
            $"Connecting | apiVersion={apiVersion} | resume={!string.IsNullOrWhiteSpace(resumptionHandle)} | search={_googleSearchEnabled} | model={settings.LiveModel}");

        _client?.Dispose();
        _client = new Client(
            apiKey: settings.ApiKey,
            httpOptions: new HttpOptions
            {
                ApiVersion = apiVersion
            });

        var config = BuildConnectConfig(settings, resumptionHandle);
        _session = await _client.Live.ConnectAsync(settings.LiveModel, config, cancellationToken);
        _connectedAt = DateTime.UtcNow;

        LogSession("Connect", "Connected.");

        _receiveLoopCts?.Dispose();
        _receiveLoopCts = CancellationTokenSource.CreateLinkedTokenSource(
            _serviceLifetimeCts?.Token ?? CancellationToken.None);
        _receiveLoopTask = Task.Run(() => ReceiveLoopAsync(_receiveLoopCts.Token));
    }

    private LiveConnectConfig BuildConnectConfig(AppSettings settings, string? resumptionHandle)
    {
        var isGemini31 = IsGemini31Live(settings.LiveModel);

        return new LiveConnectConfig
        {
            ResponseModalities = new List<Modality> { Modality.Audio },
            EnableAffectiveDialog = !isGemini31 && settings.EnableAffectiveDialog ? true : null,
            MediaResolution = ParseMediaResolution(settings.MediaResolution),
            Proactivity = !isGemini31 && settings.EnableProactiveAudio
                ? new ProactivityConfig { ProactiveAudio = true }
                : null,
            ContextWindowCompression = BuildContextWindowCompression(settings),
            ThinkingConfig = BuildThinkingConfig(settings),
            SessionResumption = new SessionResumptionConfig
            {
                Handle = resumptionHandle
            },
            SystemInstruction = new Content
            {
                Parts = new List<Part>
                {
                    new() { Text = settings.SystemPrompt }
                }
            },
            SpeechConfig = new SpeechConfig
            {
                VoiceConfig = new VoiceConfig
                {
                    PrebuiltVoiceConfig = new PrebuiltVoiceConfig
                    {
                        VoiceName = settings.Voice
                    }
                }
            },
            RealtimeInputConfig = new RealtimeInputConfig
            {
                AutomaticActivityDetection = new AutomaticActivityDetection
                {
                    Disabled = false,
                    StartOfSpeechSensitivity = isGemini31
                        ? StartSensitivity.StartSensitivityLow
                        : (StartSensitivity?)null,
                    EndOfSpeechSensitivity = isGemini31
                        ? EndSensitivity.EndSensitivityLow
                        : (EndSensitivity?)null,
                    PrefixPaddingMs = isGemini31 ? 80 : 40,
                    SilenceDurationMs = 600
                }
            },
            Tools = _googleSearchEnabled
                ? new List<Tool>
                {
                    new()
                    {
                        GoogleSearch = new GoogleSearch()
                    }
                }
                : null,
            InputAudioTranscription = new AudioTranscriptionConfig(),
            OutputAudioTranscription = new AudioTranscriptionConfig()
        };
    }

    private async Task SendRealtimeInputAsync(
        AsyncSession session,
        LiveSendRealtimeInputParameters realtimeInput,
        string inputKind,
        CancellationToken cancellationToken)
    {
        await _sendLock.WaitAsync(cancellationToken);
        try
        {
            if (!ReferenceEquals(session, _session) || !_shouldBeRunning)
            {
                return;
            }

            await session.SendRealtimeInputAsync(realtimeInput, cancellationToken);
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
            throw;
        }
        catch (Exception ex)
        {
            LogSessionException("Send", $"Realtime {inputKind} send failed.", ex);
            var queued = QueueRecovery(
                $"Realtime {inputKind} send failed: {ex.Message}",
                unexpectedClose: true,
                invalidArgumentClose: IsInvalidArgumentClose(ex));

            if (queued)
            {
                PublishStatus($"Gemini Live send failed: {ex.Message}. Recovering with a clean connection...");
            }
        }
        finally
        {
            _sendLock.Release();
        }
    }

    private async Task RestoreConversationContextAsync(
        IReadOnlyList<ConversationLogEntryRecord> restoredConversation,
        CancellationToken cancellationToken)
    {
        var session = _session;
        if (session is null)
        {
            return;
        }

        var turns = BuildRestoredConversationTurns(restoredConversation);
        if (turns.Count == 0)
        {
            return;
        }

        LogSession("Restore", $"Loading saved context | turns={turns.Count}");
        await session.SendClientContentAsync(new LiveSendClientContentParameters
        {
            Turns = turns,
            TurnComplete = false
        }, cancellationToken);
        PublishStatus("Loaded saved conversation context into Gemini Live.");
    }

    private async Task ReceiveLoopAsync(CancellationToken cancellationToken)
    {
        LogSession("Receive", "Receive loop started.");

        try
        {
            while (!cancellationToken.IsCancellationRequested)
            {
                var session = _session;
                if (session is null)
                {
                    break;
                }

                var message = await session.ReceiveAsync(cancellationToken);
                if (message is null)
                {
                    if (_shouldBeRunning)
                    {
                        LogSession("Receive", "Gemini Live websocket closed while the session should still be running.");
                        QueueRecovery(
                            "Gemini Live closed the websocket.",
                            unexpectedClose: true,
                            invalidArgumentClose: false);
                    }
                    break;
                }

                ProcessMessage(message);
            }
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
            LogSession("Receive", "Receive loop cancelled.");
        }
        catch (Exception ex)
        {
            if (_shouldBeRunning)
            {
                LogSessionException("Receive", "Receive loop failed.", ex);
                var queued = QueueRecovery(
                    $"Receive loop failed: {ex.Message}",
                    unexpectedClose: true,
                    invalidArgumentClose: IsInvalidArgumentClose(ex));

                if (queued)
                {
                    PublishStatus($"Gemini Live connection closed: {ex.Message}. Recovering with a clean connection...");
                }
            }
        }
        finally
        {
            LogSession("Receive", "Receive loop ended.");
        }
    }

    private void ProcessMessage(LiveServerMessage message)
    {
        if (message.SessionResumptionUpdate is { } resumptionUpdate)
        {
            UpdateSessionResumptionState(resumptionUpdate);
        }

        LogServerMessage(message);

        if (message.ServerContent?.InputTranscription?.Text is { Length: > 0 } inputText)
        {
            PublishInputTranscript(inputText);
        }

        var emittedOutputTranscription = false;
        if (message.ServerContent?.OutputTranscription?.Text is { Length: > 0 } outputText)
        {
            PublishOutputTranscript(outputText);
            emittedOutputTranscription = true;
        }

        if (message.ServerContent?.Interrupted == true)
        {
            var now = DateTime.UtcNow;
            if ((now - _lastInterruptedAt) >= InterruptDebounce)
            {
                _lastInterruptedAt = now;
                LogSession("Server", "Generation interrupted by detected user activity.");
                ResetOutputTranscript();
                Interrupted?.Invoke();
            }
            else
            {
                LogSession("Server", "Duplicate interrupt event suppressed.");
            }
        }

        if (message.GoAway is not null)
        {
            PublishStatus("Gemini Live requested a scheduled reconnect. Resuming with the latest handle...");
            QueueRecovery(
                "Gemini Live sent GoAway.",
                unexpectedClose: false,
                invalidArgumentClose: false);
        }

        var parts = message.ServerContent?.ModelTurn?.Parts;
        if (parts is not null)
        {
            foreach (var part in parts)
            {
                if (!emittedOutputTranscription &&
                    part.Thought != true &&
                    !string.IsNullOrWhiteSpace(part.Text))
                {
                    PublishOutputTranscript(part.Text);
                }

                if (part.InlineData?.Data is { Length: > 0 } audioBytes)
                {
                    AudioReceived?.Invoke(audioBytes);
                }
            }
        }

        if (message.ServerContent?.GenerationComplete == true || message.ServerContent?.TurnComplete == true)
        {
            CloseTranscriptTurns();
        }
    }

    private bool QueueRecovery(
        string reason,
        bool unexpectedClose,
        bool invalidArgumentClose)
    {
        if (!_shouldBeRunning || _activeSettings is null || _serviceLifetimeCts?.IsCancellationRequested == true)
        {
            return false;
        }

        if (Interlocked.CompareExchange(ref _recoveryInFlight, 1, 0) != 0)
        {
            LogSession("Recover", $"Recovery already in flight. Ignored duplicate trigger | reason={reason}");
            return false;
        }

        var isGemini31 = IsGemini31Live(_activeSettings.LiveModel);
        var forceFreshSession = unexpectedClose && isGemini31;

        if (unexpectedClose && RegisterUnexpectedClose())
        {
            _ = StopAfterRepeatedCloseAsync(reason);
            return true;
        }

        if (isGemini31 && _googleSearchEnabled &&
            (invalidArgumentClose || (unexpectedClose && _rapidCloseCount >= 2)))
        {
            _googleSearchEnabled = false;
            forceFreshSession = true;
            ClearResumptionHandle();
            LogSession(
                "Compat",
                "Google Search disabled for this live session after repeated/invalid-argument Gemini 3.1 closes. This prevents a search-triggered resume loop. Restart Live to re-enable search.");
            PublishStatus(
                "Gemini 3.1 closed during this session. Retrying fresh with Google Search temporarily disabled to prevent the close/reconnect loop.");
        }

        if (forceFreshSession)
        {
            ClearResumptionHandle();
        }

        _ = RecoverSessionAsync(reason, forceFreshSession);
        return true;
    }

    private async Task RecoverSessionAsync(string reason, bool forceFreshSession)
    {
        var reconnectLockHeld = false;
        try
        {
            var settings = _activeSettings;
            if (!_shouldBeRunning || settings is null || _serviceLifetimeCts?.IsCancellationRequested == true)
            {
                return;
            }

            var lifetimeToken = _serviceLifetimeCts.Token;
            await _reconnectLock.WaitAsync(lifetimeToken);
            reconnectLockHeld = true;

            await DisposeCurrentSessionAsync(resetTranscripts: false, CancellationToken.None);
            lifetimeToken.ThrowIfCancellationRequested();

            for (var attempt = 1; attempt <= MaxReconnectAttempts; attempt++)
            {
                try
                {
                    var canResume = !forceFreshSession && !string.IsNullOrWhiteSpace(GetResumptionHandle());
                    PublishStatus(attempt == 1
                        ? $"Recovering Gemini Live: {reason}"
                        : $"Reconnect attempt {attempt}/{MaxReconnectAttempts}: {reason}");
                    LogSession(
                        "Recover",
                        $"Reconnect attempt {attempt} | fresh={forceFreshSession} | resume={canResume} | search={_googleSearchEnabled}");

                    await ConnectAsync(settings, allowResumption: canResume, lifetimeToken);
                    PublishStatus(canResume
                        ? "Gemini Live reconnected and resumed."
                        : "Gemini Live reconnected with a clean session.");
                    return;
                }
                catch (OperationCanceledException) when (lifetimeToken.IsCancellationRequested)
                {
                    return;
                }
                catch (Exception ex)
                {
                    LogSessionException("Recover", $"Reconnect attempt {attempt} failed.", ex);

                    if (IsGemini31Live(settings.LiveModel) &&
                        _googleSearchEnabled &&
                        IsInvalidArgumentClose(ex))
                    {
                        _googleSearchEnabled = false;
                        forceFreshSession = true;
                        ClearResumptionHandle();
                        LogSession("Compat", "Reconnect got invalid argument; retrying Gemini 3.1 without Google Search and without resuming the rejected session.");
                    }

                    if (attempt >= MaxReconnectAttempts)
                    {
                        await StopAfterReconnectFailureAsync(ex.Message);
                        return;
                    }

                    await Task.Delay(Math.Min(5000, 750 * attempt), lifetimeToken);
                }
            }
        }
        catch (OperationCanceledException)
        {
        }
        finally
        {
            if (reconnectLockHeld)
            {
                _reconnectLock.Release();
            }

            Interlocked.Exchange(ref _recoveryInFlight, 0);
        }
    }

    private bool RegisterUnexpectedClose()
    {
        var now = DateTime.UtcNow;
        var age = _connectedAt == DateTime.MinValue
            ? TimeSpan.Zero
            : now - _connectedAt;

        if (age <= RapidCloseWindow)
        {
            _rapidCloseCount++;
        }
        else
        {
            _rapidCloseCount = 1;
        }

        LogSession(
            "Recover",
            $"Unexpected close | connectionAge={age.TotalSeconds:0.0}s | rapidCloseCount={_rapidCloseCount}/{RapidCloseLimit}");

        return _rapidCloseCount >= RapidCloseLimit;
    }

    private async Task StopAfterRepeatedCloseAsync(string reason)
    {
        try
        {
            PublishStatus(
                "Gemini Live closed repeatedly. Auto-reconnect was stopped instead of looping. Press Start Session to try again.");
            LogSession("Recover", $"Circuit breaker opened after repeated closes | reason={reason}");

            _shouldBeRunning = false;
            _serviceLifetimeCts?.Cancel();
            await DisposeCurrentSessionAsync(resetTranscripts: false, CancellationToken.None);
            _diagnosticsLogService.EndSession("Repeated Gemini Live closes");
            SessionStateChanged?.Invoke(false);
        }
        finally
        {
            Interlocked.Exchange(ref _recoveryInFlight, 0);
        }
    }

    private async Task StopAfterReconnectFailureAsync(string message)
    {
        PublishStatus(
            $"Gemini Live could not reconnect after {MaxReconnectAttempts} attempts: {message}. Session stopped.");
        _shouldBeRunning = false;
        _serviceLifetimeCts?.Cancel();
        await DisposeCurrentSessionAsync(resetTranscripts: false, CancellationToken.None);
        _diagnosticsLogService.EndSession("Reconnect failed");
        SessionStateChanged?.Invoke(false);
    }

    private async Task DisposeCurrentSessionAsync(bool resetTranscripts, CancellationToken cancellationToken)
    {
        _receiveLoopCts?.Cancel();

        var session = _session;
        _session = null;

        if (session is not null)
        {
            try
            {
                await session.CloseAsync();
            }
            catch (Exception ex)
            {
                LogSession("Dispose", $"Session close ignored: {ex.Message}");
            }

            try
            {
                await session.DisposeAsync();
            }
            catch (Exception ex)
            {
                LogSession("Dispose", $"Session dispose ignored: {ex.Message}");
            }
        }

        if (_receiveLoopTask is not null)
        {
            try
            {
                await _receiveLoopTask.WaitAsync(cancellationToken);
            }
            catch
            {
            }
            _receiveLoopTask = null;
        }

        _receiveLoopCts?.Dispose();
        _receiveLoopCts = null;
        _client?.Dispose();
        _client = null;

        if (resetTranscripts)
        {
            ResetTranscriptState();
        }

        LogSession("Dispose", $"Disposed current session | resetTranscripts={resetTranscripts}");
    }

    private void UpdateSessionResumptionState(LiveServerSessionResumptionUpdate update)
    {
        lock (_resumptionLock)
        {
            if (update.Resumable == true && !string.IsNullOrWhiteSpace(update.NewHandle))
            {
                _sessionResumptionHandle = update.NewHandle;
            }
            else if (update.Resumable == false)
            {
                _sessionResumptionHandle = null;
            }
        }

        LogSession(
            "Resumption",
            $"Update | resumable={update.Resumable} | handlePresent={!string.IsNullOrWhiteSpace(update.NewHandle)}");
    }

    private string? GetResumptionHandle()
    {
        lock (_resumptionLock)
        {
            return _sessionResumptionHandle;
        }
    }

    private void ClearResumptionHandle()
    {
        lock (_resumptionLock)
        {
            _sessionResumptionHandle = null;
        }
    }

    private void ResetConnectionState()
    {
        ClearResumptionHandle();
        _googleSearchEnabled = true;
        _connectedAt = DateTime.MinValue;
        _rapidCloseCount = 0;
        _lastInterruptedAt = DateTime.MinValue;
        Interlocked.Exchange(ref _recoveryInFlight, 0);
    }

    private void PublishInputTranscript(string chunk)
    {
        var aggregated = MergeTranscriptChunk(
            chunk,
            ref _currentInputTranscript,
            ref _lastInputTranscriptAt,
            ref _inputTurnOpen);

        if (aggregated is not null)
        {
            InputTranscriptionChanged?.Invoke(NormalizeTranscriptForDisplay(aggregated));
        }
    }

    private void PublishOutputTranscript(string chunk)
    {
        var aggregated = MergeTranscriptChunk(
            chunk,
            ref _currentOutputTranscript,
            ref _lastOutputTranscriptAt,
            ref _outputTurnOpen);

        if (aggregated is not null)
        {
            lock (_transcriptLock)
            {
                _inputTurnOpen = false;
            }

            OutputTranscriptionChanged?.Invoke(NormalizeTranscriptForDisplay(aggregated));
        }
    }

    private string? MergeTranscriptChunk(
        string chunk,
        ref string currentTranscript,
        ref DateTime lastUpdatedAt,
        ref bool turnOpen)
    {
        var cleaned = chunk.Trim();
        if (string.IsNullOrWhiteSpace(cleaned))
        {
            return null;
        }

        lock (_transcriptLock)
        {
            var now = DateTime.UtcNow;
            var shouldReset = !turnOpen || (now - lastUpdatedAt) > TimeSpan.FromSeconds(3);
            if (shouldReset)
            {
                currentTranscript = cleaned;
                turnOpen = true;
            }
            else
            {
                currentTranscript = MergeIncrementalText(currentTranscript, cleaned);
            }

            lastUpdatedAt = now;
            return currentTranscript;
        }
    }

    private static string MergeIncrementalText(string existing, string incoming)
    {
        if (string.IsNullOrWhiteSpace(existing))
        {
            return incoming;
        }

        if (string.Equals(existing, incoming, StringComparison.Ordinal))
        {
            return existing;
        }

        if (incoming.StartsWith(existing, StringComparison.Ordinal))
        {
            return incoming;
        }

        if (existing.StartsWith(incoming, StringComparison.Ordinal) || existing.EndsWith(incoming, StringComparison.Ordinal))
        {
            return existing;
        }

        if (incoming.Length > 0 && ".,!?;:)]}".Contains(incoming[0]))
        {
            return existing + incoming;
        }

        if (existing.Length > 0 && "([{".Contains(existing[^1]))
        {
            return existing + incoming;
        }

        return existing + " " + incoming;
    }

    private static List<Content> BuildRestoredConversationTurns(
        IReadOnlyList<ConversationLogEntryRecord> restoredConversation)
    {
        var turns = new List<Content>();
        Content? current = null;
        string? currentRole = null;

        foreach (var entry in restoredConversation)
        {
            var text = entry.Text?.Trim();
            if (string.IsNullOrWhiteSpace(text))
            {
                continue;
            }

            var role = MapSavedConversationRole(entry.Role);
            if (role is null)
            {
                continue;
            }

            if (!string.Equals(currentRole, role, StringComparison.Ordinal))
            {
                current = new Content
                {
                    Role = role,
                    Parts = new List<Part>()
                };
                turns.Add(current);
                currentRole = role;
            }

            current!.Parts ??= new List<Part>();
            current.Parts.Add(new Part { Text = text });
        }

        return turns;
    }

    private static string? MapSavedConversationRole(string? role)
    {
        if (string.Equals(role, "You", StringComparison.OrdinalIgnoreCase))
        {
            return "user";
        }

        if (string.Equals(role, "Gemini", StringComparison.OrdinalIgnoreCase))
        {
            return "model";
        }

        return null;
    }

    private static string NormalizeTranscriptForDisplay(string transcript)
    {
        if (string.IsNullOrWhiteSpace(transcript))
        {
            return transcript;
        }

        var normalized = PunctuationSpacingRegex.Replace(transcript, " ");
        normalized = DuplicateWhitespaceRegex.Replace(normalized, " ");
        return normalized.Trim();
    }

    private void CloseTranscriptTurns()
    {
        lock (_transcriptLock)
        {
            _inputTurnOpen = false;
            _outputTurnOpen = false;
        }
    }

    private void ResetTranscriptState()
    {
        lock (_transcriptLock)
        {
            _currentInputTranscript = string.Empty;
            _currentOutputTranscript = string.Empty;
            _lastInputTranscriptAt = DateTime.MinValue;
            _lastOutputTranscriptAt = DateTime.MinValue;
            _inputTurnOpen = false;
            _outputTurnOpen = false;
        }
    }

    private void ResetOutputTranscript()
    {
        lock (_transcriptLock)
        {
            _currentOutputTranscript = string.Empty;
            _lastOutputTranscriptAt = DateTime.MinValue;
            _outputTurnOpen = false;
        }
    }

    private static bool IsGemini31Live(string? model)
    {
        return !string.IsNullOrWhiteSpace(model) &&
               model.Contains("gemini-3.1-flash-live", StringComparison.OrdinalIgnoreCase);
    }

    private static bool IsThinkingDisabled(AppSettings settings)
    {
        return settings.EnableThinkingConfig &&
               string.Equals(settings.ThinkingMode, "disabled", StringComparison.OrdinalIgnoreCase);
    }

    private static bool RequiresV1Alpha(AppSettings settings)
    {
        if (IsGemini31Live(settings.LiveModel))
        {
            return false;
        }

        return settings.EnableAffectiveDialog || settings.EnableProactiveAudio;
    }

    private static ContextWindowCompressionConfig? BuildContextWindowCompression(AppSettings settings)
    {
        if (!settings.EnableContextWindowCompression)
        {
            return null;
        }

        var config = new ContextWindowCompressionConfig
        {
            TriggerTokens = Math.Max(1, settings.ContextCompressionTriggerTokens),
            SlidingWindow = new SlidingWindow()
        };

        if (settings.ContextCompressionTargetTokens > 0)
        {
            config.SlidingWindow.TargetTokens = settings.ContextCompressionTargetTokens;
        }

        return config;
    }

    private static ThinkingConfig? BuildThinkingConfig(AppSettings settings)
    {
        var mode = settings.ThinkingMode?.Trim().ToLowerInvariant();
        if (!settings.EnableThinkingConfig || mode == "default")
        {
            return null;
        }

        if (IsGemini31Live(settings.LiveModel))
        {
            var level = mode == "disabled"
                ? ThinkingLevel.Minimal
                : ParseThinkingLevel(settings.ThinkingLevel) ?? ThinkingLevel.Minimal;

            return new ThinkingConfig
            {
                IncludeThoughts = settings.IncludeThoughts ? true : null,
                ThinkingLevel = level
            };
        }

        if (mode == "disabled")
        {
            return new ThinkingConfig
            {
                ThinkingBudget = 0
            };
        }

        return new ThinkingConfig
        {
            IncludeThoughts = settings.IncludeThoughts,
            ThinkingBudget = settings.ThinkingBudget,
            ThinkingLevel = ParseThinkingLevel(settings.ThinkingLevel)
        };
    }

    private static MediaResolution? ParseMediaResolution(string? value)
    {
        return value?.Trim().ToLowerInvariant() switch
        {
            "default" or "" => (MediaResolution?)null,
            "low" => MediaResolution.MediaResolutionLow,
            "medium" => MediaResolution.MediaResolutionMedium,
            "high" => MediaResolution.MediaResolutionHigh,
            _ => MediaResolution.MediaResolutionLow
        };
    }

    private static ThinkingLevel? ParseThinkingLevel(string? value)
    {
        return value?.Trim().ToLowerInvariant() switch
        {
            "default" or "" => (ThinkingLevel?)null,
            "minimal" => ThinkingLevel.Minimal,
            "low" => ThinkingLevel.Low,
            "medium" => ThinkingLevel.Medium,
            "high" => ThinkingLevel.High,
            _ => (ThinkingLevel?)null
        };
    }

    private static bool IsInvalidArgumentClose(Exception ex)
    {
        return ex.Message.Contains(InvalidArgumentMarker, StringComparison.OrdinalIgnoreCase) ||
               ex.Message.Contains("InvalidPayloadData", StringComparison.OrdinalIgnoreCase) ||
               ex.InnerException is not null && IsInvalidArgumentClose(ex.InnerException);
    }

    private void PublishStatus(string message)
    {
        LogSession("Status", message);
        StatusChanged?.Invoke(message);
    }

    private void LogServerMessage(LiveServerMessage message)
    {
        var modelParts = message.ServerContent?.ModelTurn?.Parts;
        var textParts = modelParts?.Count(part => !string.IsNullOrWhiteSpace(part.Text) && part.Thought != true) ?? 0;
        var thoughtParts = modelParts?.Count(part => part.Thought == true) ?? 0;
        var audioParts = modelParts?.Count(part => part.InlineData?.Data is { Length: > 0 }) ?? 0;

        LogSession(
            "Server",
            $"Message | goAway={message.GoAway is not null} | interrupted={message.ServerContent?.Interrupted == true} | turnComplete={message.ServerContent?.TurnComplete == true} | generationComplete={message.ServerContent?.GenerationComplete == true} | inputText={(message.ServerContent?.InputTranscription?.Text?.Length ?? 0)} chars | outputText={(message.ServerContent?.OutputTranscription?.Text?.Length ?? 0)} chars | textParts={textParts} | thoughtParts={thoughtParts} | audioParts={audioParts}");
    }

    private void LogSession(string category, string message)
    {
        _diagnosticsLogService.LogSession(category, message);
    }

    private void LogSessionException(string category, string message, Exception exception)
    {
        _diagnosticsLogService.LogSessionException(category, message, exception);
    }
}
