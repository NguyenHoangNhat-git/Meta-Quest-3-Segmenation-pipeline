using UnityEngine;
using UnityEngine.UI;

namespace CVMQ3.ProcessingPipeline
{
    public class ObjectInfoSelection : MonoBehaviour
    {
        [Header("References")]
        [SerializeField]
        private HeadGazeSelector m_gazeSelector;

        [SerializeField]
        private TextAsset m_objectInfoJson;

        [SerializeField]
        private GameObject m_selectionPanel;

        [SerializeField]
        private Text m_infoText;

        [Header("Text Style")]
        [SerializeField]
        private Color m_textColor = Color.white;

        [SerializeField]
        private int m_fontSize = 12;

        [SerializeField]
        private Font m_font;

        [Header("Panel Size")]
        [Tooltip("Fixed world-space width of the info panel.")]
        [SerializeField]
        private float m_panelWorldWidth = 0.18f;

        [Tooltip("Fixed world-space height of the info panel.")]
        [SerializeField]
        private float m_panelWorldHeight = 0.22f;

        [Tooltip("Extra gap between the right edge of the bounding box and the panel.")]
        [SerializeField]
        private float m_horizontalGap = 0.02f;
        [Tooltip("Padding")]
        [SerializeField]
        private float m_padding = 10f;

        [Header("Layout")]
        [SerializeField]
        private float m_followSpeed = 8f;

        private ObjectInfoDatabase m_db;
        private MultiDetectionUiManager.BoundingBoxData m_target;
        private RectTransform m_panelRect;
        private float m_canvasScale = 1f;

        // ── Lifecycle ────────────────────────────────────────────────────────

        private void Awake()
        {
            if (m_infoText == null)
            {
                Debug.LogError("[ObjInfo] No Text component found in children.");
            }

            // Cache the panel's RectTransform so we can resize it at runtime.
            if (m_selectionPanel != null)
                m_panelRect = m_selectionPanel.GetComponent<RectTransform>();

            ApplyTextStyle();

            if (m_objectInfoJson != null)
            {
                m_db = ObjectInfoDatabase.Load(m_objectInfoJson);
                // ── Debug: confirm how many entries actually loaded ──────────
                if (m_db == null)
                    Debug.LogError("[ObjInfo] DB is null after Load().");
                else
                    Debug.Log("[ObjInfo] DB loaded OK.");
            }
            else
            {
                Debug.LogWarning("[ObjInfo] No object_info.json assigned.");
            }

            m_selectionPanel.SetActive(false);
            var canvas = m_selectionPanel.GetComponentInParent<Canvas>();
            if (canvas != null)
                m_canvasScale = canvas.transform.lossyScale.x; // assumes uniform scale
            Debug.Log($"[ObjInfo] Canvas lossyScale = {canvas.transform.lossyScale}");
        }

        private void OnEnable()
        {
            if (m_gazeSelector == null)
            {
                Debug.LogError("[ObjInfo] HeadGazeSelector not assigned.");
                return;
            }
            m_gazeSelector.OnSelectionChanged += HandleSelectionChanged;
        }

        private void OnDisable()
        {
            if (m_gazeSelector != null)
                m_gazeSelector.OnSelectionChanged -= HandleSelectionChanged;
            m_target = null;
            m_selectionPanel.SetActive(false);
        }

        private void Update()
        {
            if (m_target?.BoxRectTransform == null)
                return;

            Vector3 targetPos = ComputePanelPosition(m_target);
            transform.position = Vector3.Lerp(
                transform.position,
                targetPos,
                Time.deltaTime * m_followSpeed
            );

            // Billboard: always face the camera.
            var cam = Camera.main;
            if (cam != null)
                transform.rotation = Quaternion.LookRotation(
                    transform.position - cam.transform.position
                );
        }

        // ── Helpers ──────────────────────────────────────────────────────────

        /// <summary>
        /// Computes the world-space position that places the panel flush to the
        /// right edge of the bounding box with a small gap.
        /// </summary>
        private Vector3 ComputePanelPosition(MultiDetectionUiManager.BoundingBoxData box)
        {
            var rt = box.BoxRectTransform;
            float boxHalfWidth = rt.sizeDelta.x * rt.lossyScale.x * 0.5f;

            // Panel half-width in world space — use the world meters value directly
            float panelHalfWidth = m_panelWorldWidth * 0.5f;

            Vector3 rightDir = rt.right;
            return rt.position + rightDir * (boxHalfWidth + m_horizontalGap + panelHalfWidth);
        }

        public void ApplyTextStyle()
        {
            if (m_infoText == null)
                return;

            m_infoText.color = m_textColor;
            m_infoText.fontSize = m_fontSize;
            m_infoText.supportRichText = true;
            m_infoText.resizeTextForBestFit = false; // don't let Unity override your size

            // Make sure the text rect fills the panel
            var textRect = m_infoText.GetComponent<RectTransform>();
            if (textRect != null)
            {
                textRect.anchorMin = Vector2.zero;
                textRect.anchorMax = Vector2.one;
                textRect.offsetMin = new Vector2(m_padding, m_padding); // small padding in local units
                textRect.offsetMax = new Vector2(-m_padding, -m_padding);
            }

            m_infoText.horizontalOverflow = HorizontalWrapMode.Wrap;
            m_infoText.verticalOverflow = VerticalWrapMode.Truncate; // or Truncate

            if (m_font != null)
                m_infoText.font = m_font;
        }

        private void SizePanelToConfig()
        {
            if (m_panelRect == null || m_canvasScale == 0f)
                return;
            float localWidth = m_panelWorldWidth / m_canvasScale;
            float localHeight = m_panelWorldHeight / m_canvasScale;
            m_panelRect.sizeDelta = new Vector2(localWidth, localHeight);
            Debug.Log($"[ObjInfo] Panel sizeDelta set to {m_panelRect.sizeDelta}");
        }

        // ── Selection handler ────────────────────────────────────────────────

        private void HandleSelectionChanged(MultiDetectionUiManager.BoundingBoxData selected)
        {
            if (selected == null)
            {
                m_target = null;
                m_selectionPanel.SetActive(false);
                return;
            }

            m_target = selected;

            // Activate and size FIRST
            SizePanelToConfig();
            transform.position = ComputePanelPosition(selected);
            m_selectionPanel.SetActive(true);

            // Look up info
            ObjectInfoEntry info = null;
            bool found = m_db != null && m_db.TryGet(selected.ClassId, out info);

            // Set text LAST, after the panel is active and layout is settled
            if (m_infoText != null)
            {
                m_infoText.text = found
                    ? BuildInfoString(info)
                    : $"<b>{selected.ClassName}</b>\n(no info available)";

                UnityEngine.UI.LayoutRebuilder.ForceRebuildLayoutImmediate(m_panelRect);
            }
        }

        // ── Formatting ───────────────────────────────────────────────────────

        private static string BuildInfoString(ObjectInfoEntry info)
        {
            var sb = new System.Text.StringBuilder();

            sb.AppendLine($"<b>{info.name}</b>");

            if (info.aliases is { Count: > 0 })
                sb.AppendLine($"<i>{string.Join(", ", info.aliases)}</i>");

            sb.AppendLine($"Category: {info.category} / {info.supercategory}");

            if (info.material is { Count: > 0 })
            {
                sb.AppendLine($"Material: {string.Join(", ", info.material)}");
            }

            if (info.shape_profile != null)
                sb.AppendLine(
                    $"Shape: {info.shape_profile.geometry} ({info.shape_profile.symmetry})"
                );

            if (info.typical_dimensions_cm != null)
            {
                var d = info.typical_dimensions_cm;
                if (d.height is { Count: 2 })
                    sb.AppendLine($"Height: {d.height[0]}–{d.height[1]} cm");
                if (d.diameter is { Count: 2 })
                    sb.AppendLine($"Diameter: {d.diameter[0]}–{d.diameter[1]} cm");
            }

            if (info.segmentation != null)
            {
                sb.AppendLine($"\nSeg difficulty: <b>{info.segmentation.difficulty}</b>");
                if (!string.IsNullOrEmpty(info.segmentation.reason))
                    sb.AppendLine($"  {info.segmentation.reason}");
                if (!string.IsNullOrEmpty(info.segmentation.recommended_approach))
                    sb.AppendLine($"  Approach: {info.segmentation.recommended_approach}");
            }

            if (!string.IsNullOrEmpty(info.occlusion_risk))
                sb.AppendLine($"\nOcclusion risk: {info.occlusion_risk}");

            if (!string.IsNullOrEmpty(info.pose_estimation_notes))
                sb.AppendLine($"\n{info.pose_estimation_notes}");

            Debug.Log($"[ObjInfo] Finished building info: {sb.ToString().TrimEnd()}");
            return sb.ToString().TrimEnd();
        }

#if UNITY_EDITOR
        private void OnValidate()
        {
            if (m_infoText == null)
                m_infoText = GetComponentInChildren<Text>(includeInactive: true);
            // Only restyle — never assign .text here
            ApplyTextStyle();
        }
#endif
    }
}
