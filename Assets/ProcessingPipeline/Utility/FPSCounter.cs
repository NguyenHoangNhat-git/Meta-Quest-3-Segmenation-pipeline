using CVMQ3.ProcessingPipeline;
using TMPro;
using UnityEngine;

public class FPSCounter : MonoBehaviour
{
    [SerializeField]
    private TextMeshProUGUI m_fpsText;

    [SerializeField]
    private float m_updateInterval = 0.1f;

    [Header("Optional — assign to show inference timing")]
    [SerializeField]
    private MultiDetectionRunManager m_detectManager;

    [SerializeField]
    private SegmentationRunManager m_segManager;

    [SerializeField]
    private GazeCropExtractor m_cropExtractor;

    // FPS tracking
    private float m_accumulatedTime;
    private int m_frameCount;
    private float m_timeLeft;

    // Detection timing
    private float m_lastDetectStart;
    private float m_lastDetectDuration;
    private float m_detectUpdateTime;

    // Segmentation pipeline timing (crop-ready → segmentation output)
    private float m_lastSegStart;
    private float m_lastSegDuration;
    private float m_segUpdateTime;

    private float m_cachedCropInterval;

    void Start()
    {
        if (m_fpsText == null)
            m_fpsText = GetComponent<TextMeshProUGUI>();

        m_timeLeft = m_updateInterval;
        m_accumulatedTime = 0f;
        m_frameCount = 0;

        if (m_detectManager != null)
        {
            m_detectManager.OnInferenceStart += () => m_lastDetectStart = Time.unscaledTime;
            m_detectManager.OnInferenceComplete += () =>
            {
                m_lastDetectDuration = Time.unscaledTime - m_lastDetectStart;
                m_detectUpdateTime = Time.unscaledTime;
            };
        }

        if (m_cropExtractor != null)
            m_cropExtractor.OnCropReady += OnCropReady;

        if (m_segManager != null)
        {
            // Two-parameter signature matches updated Action<SegResult, Pose> — no proto tensor.
            m_segManager.OnSegmentationReady += (result, pose) =>
            {
                m_lastSegDuration = Time.unscaledTime - m_lastSegStart;
                m_segUpdateTime = Time.unscaledTime;
            };
        }

        // m_extractInterval is internal so we can access it directly — no reflection needed.
        m_cachedCropInterval = m_cropExtractor != null ? m_cropExtractor.m_extractInterval : 0f;
    }

    void Update()
    {
        m_timeLeft -= Time.unscaledDeltaTime;
        m_accumulatedTime += Time.unscaledDeltaTime;
        m_frameCount++;

        if (m_timeLeft > 0f)
            return;

        float fps = m_frameCount / m_accumulatedTime;
        float detectAge = m_detectUpdateTime > 0 ? Time.unscaledTime - m_detectUpdateTime : -1f;
        float segAge = m_segUpdateTime > 0 ? Time.unscaledTime - m_segUpdateTime : -1f;

        m_fpsText.text =
            $"{fps:F0} FPS\n"
            + $"Detect:   {m_lastDetectDuration * 1000:F0}ms"
            + (detectAge >= 0 ? $"({detectAge:F1}s ago)" : "(none)")
            + "\n\n"
            + $"Segment:  {m_lastSegDuration * 1000:F0}ms  "
            + (segAge >= 0 ? $"({segAge:F1}s ago)" : "(none)")
            + "\n\n"
            + $"CropInterval: {m_cachedCropInterval:F2}s";

        // Quest 3 dynamic refresh rate tiers: 72 / 80 / 90 Hz
        m_fpsText.color =
            fps < 72 ? Color.red
            : fps < 89 ? Color.yellow
            : Color.green;

        m_timeLeft += m_updateInterval; // += preserves overshoot correction
        m_accumulatedTime = 0f;
        m_frameCount = 0;
    }

    // Single overload matching Action<RenderTexture, CropInfo> — old Texture2D overload removed.
    private void OnCropReady(RenderTexture _, CropInfo __)
    {
        m_lastSegStart = Time.unscaledTime;
    }

    void OnDestroy()
    {
        if (m_cropExtractor != null)
            m_cropExtractor.OnCropReady -= OnCropReady;
    }
}
