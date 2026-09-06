using System.IO;
using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Windows;
using WpfComboBox = System.Windows.Controls.ComboBox;

namespace WinAiBuddy;

internal static class LiveModelPickerBootstrap
{
    private const string RecommendedModel = "gemini-3.1-flash-live-preview";

    private static readonly string[] KnownLiveModels =
    [
        RecommendedModel,
        "gemini-2.5-flash-native-audio-preview-12-2025"
    ];

    private static readonly ConditionalWeakTable<WpfComboBox, object> ConfiguredPickers = new();

    [ModuleInitializer]
    internal static void Initialize()
    {
        EventManager.RegisterClassHandler(
            typeof(MainWindow),
            FrameworkElement.LoadedEvent,
            new RoutedEventHandler(OnMainWindowLoaded));
    }

    private static void OnMainWindowLoaded(object sender, RoutedEventArgs e)
    {
        if (sender is not MainWindow window ||
            window.FindName("LiveModelComboBox") is not WpfComboBox comboBox ||
            ConfiguredPickers.TryGetValue(comboBox, out _))
        {
            return;
        }

        ConfiguredPickers.Add(comboBox, new object());

        var savedModel = ReadSavedModel();
        var currentModel = !string.IsNullOrWhiteSpace(savedModel)
            ? savedModel
            : RecommendedModel;

        var models = KnownLiveModels.ToList();
        if (!models.Any(model => string.Equals(model, currentModel, StringComparison.OrdinalIgnoreCase)))
        {
            models.Insert(0, currentModel.Trim());
        }

        comboBox.ItemsSource = models;
        comboBox.IsEditable = true;
        comboBox.IsTextSearchEnabled = true;
        comboBox.StaysOpenOnEdit = true;
        comboBox.ToolTip =
            $"Recommended: {RecommendedModel}. You can also type another Gemini Live model ID.";

        var selectedModel = models.FirstOrDefault(model =>
                string.Equals(model, currentModel, StringComparison.OrdinalIgnoreCase))
            ?? RecommendedModel;

        comboBox.SelectedItem = selectedModel;
        comboBox.Text = selectedModel;
        comboBox.LostKeyboardFocus += (_, _) => CommitTypedModel(comboBox);
    }

    private static void CommitTypedModel(WpfComboBox comboBox)
    {
        var typedModel = comboBox.Text?.Trim();
        if (string.IsNullOrWhiteSpace(typedModel))
        {
            return;
        }

        var models = comboBox.Items
            .Cast<object>()
            .Select(item => item?.ToString())
            .Where(item => !string.IsNullOrWhiteSpace(item))
            .Select(item => item!)
            .ToList();

        var existingModel = models.FirstOrDefault(model =>
            string.Equals(model, typedModel, StringComparison.OrdinalIgnoreCase));

        if (existingModel is null)
        {
            models.Insert(0, typedModel);
            comboBox.ItemsSource = models;
            existingModel = typedModel;
        }

        comboBox.SelectedItem = existingModel;
        comboBox.Text = existingModel;
    }

    private static string ReadSavedModel()
    {
        try
        {
            var settingsPath = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                "WinAiBuddy",
                "appsettings.json");

            if (!File.Exists(settingsPath))
            {
                return string.Empty;
            }

            using var document = JsonDocument.Parse(File.ReadAllText(settingsPath));
            if (document.RootElement.TryGetProperty("LiveModel", out var liveModelElement) &&
                liveModelElement.ValueKind == JsonValueKind.String)
            {
                return liveModelElement.GetString()?.Trim() ?? string.Empty;
            }
        }
        catch
        {
            // Keep the recommended default if the settings file cannot be read.
        }

        return string.Empty;
    }
}
