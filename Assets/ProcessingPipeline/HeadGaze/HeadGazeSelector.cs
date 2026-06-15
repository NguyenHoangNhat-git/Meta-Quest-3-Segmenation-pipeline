using UnityEngine;
using UnityEngine.UI;

namespace CVMQ3.ProcessingPipeline
{
    public class HeadGazeSelector : MonoBehaviour
    {
        [Header("Scene References")]
        [SerializeField]
        private EnvironmentRayCastSampleManager m_environmentRaycast;

        [SerializeField]
        private MultiDetectionUiManager m_uiManager;

        [Header("Gaze Reticle")]
        [Tooltip("The sphere GameObject parented under CenterEyeAnchor.")]
        [SerializeField]
        private Transform m_reticle;

        [Tooltip("How far forward (local Z) the reticle sits from the camera.")]
        [SerializeField]
        private float m_reticleDistance = 2f;

        [SerializeField]
        private Transform m_centerEyeAnchor;

        [Header("Selection")]
        [SerializeField]
        private Color m_defaultBoxColor = Color.white;

        [SerializeField]
        private Color m_selectedBoxColor = Color.red;

        private MultiDetectionUiManager.BoundingBoxData m_currentlySelected;
        public System.Action<MultiDetectionUiManager.BoundingBoxData> OnSelectionChanged;

        private void Awake()
        {
            if (m_reticle == null)
                Debug.LogError(
                    "[HeadGaze] Reticle not assigned. "
                        + "Drag the sphere (child of CenterEyeAnchor) here."
                );
            if (m_uiManager == null)
                Debug.LogError("[HeadGaze] MultiDetectionUiManager not assigned.");
            if (m_environmentRaycast == null)
                Debug.LogWarning(
                    "[HeadGaze] EnvironmentRaycast not assigned — "
                        + "reticle will sit at fixed distance."
                );
        }

        private void Update()
        {
            if (m_reticle == null || m_uiManager == null || m_centerEyeAnchor == null)
                return;

            m_reticle.localPosition = Vector3.forward * m_reticleDistance;

            // Ray from head through the reticle sphere — extending infinitely forward
            Vector3 gazeOrigin = m_centerEyeAnchor.position;
            Vector3 gazeDir = m_centerEyeAnchor.forward;

            // Debug.Log($"[HeadGaze] Origin: {gazeOrigin:F2} | Dir: {gazeDir:F2}");

            SelectBoxOnRay(gazeOrigin, gazeDir);
        }

        private void OnDestroy() => ClearSelection();

        // ── Selection ────────────────────────────────────────────────────────

        private void SelectBoxOnRay(Vector3 gazeOrigin, Vector3 gazeDir)
        {
            var boxes = m_uiManager.m_boxDrawn;

            // Debug.Log($"[HeadGaze] Boxes: {boxes?.Count ?? -1} | Origin: {gazeOrigin:F2} | Dir: {gazeDir:F2}");

            if (boxes == null || boxes.Count == 0)
            {
                // Debug.Log("[HeadGaze] No boxes to check.");
                ClearSelection();
                return;
            }

            MultiDetectionUiManager.BoundingBoxData best = null;
            float bestAngle = float.MaxValue;
            string closestInfo = "";

            for (int i = 0; i < boxes.Count; i++)
            {
                var box = boxes[i];
                if (box?.BoxRectTransform == null)
                {
                    // Debug.Log($"[HeadGaze] Box {i}: RectTransform is null!");
                    continue;
                }

                RectTransform rt = box.BoxRectTransform;
                Vector3 boxCenter = rt.position;

                // Vector from gaze origin to box center
                Vector3 toBox = boxCenter - gazeOrigin;

                // Project box center onto the gaze ray
                float projDist = Vector3.Dot(toBox, gazeDir);
                if (projDist < 0f)
                {
                    // Debug.Log($"[HeadGaze] Box {i}: Behind camera. projDist={projDist:F2}");
                    continue;
                }

                Vector3 closestPoint = gazeOrigin + gazeDir * projDist;

                // World-space distance from the ray to the box center
                float rayToBoxDist = Vector3.Distance(closestPoint, boxCenter);

                // Convert to an angular threshold based on the box's actual world size
                // sizeDelta is in local units — we need world size
                // The box scale gives us the conversion: world size = localSize * scale
                Vector3 worldScale = rt.lossyScale;
                float worldWidth = rt.sizeDelta.x * worldScale.x;
                float worldHeight = rt.sizeDelta.y * worldScale.y;

                // Use half the diagonal as the hit radius — generous but accurate
                float hitRadius =
                    Mathf.Sqrt(worldWidth * worldWidth + worldHeight * worldHeight) * 0.5f;

                // Debug.Log($"[HeadGaze] Box {i}: center={boxCenter:F2} projDist={projDist:F2}");
                // Debug.Log($"Box {i}: rayDist={rayToBoxDist:F2} hitRadius={hitRadius:F2} " +
                //             $"size={worldWidth:F2}x{worldHeight:F2} scale={worldScale:F2}");

                // Check if the ray passes within the box bounds
                if (rayToBoxDist > hitRadius)
                {
                    closestInfo =
                        $"Box {i}: MISS (rayDist {rayToBoxDist:F2} > radius {hitRadius:F2})";
                    continue;
                }

                // Among all hit boxes pick the one whose center is most aligned
                // with the gaze ray (smallest angle = most centered in view)
                float angle = Vector3.Angle(gazeDir, toBox.normalized);
                if (angle < bestAngle)
                {
                    bestAngle = angle;
                    best = box;
                    closestInfo = $"Box {i}: HIT angle={angle:F2}";
                }
            }

            // Debug.Log($"[HeadGaze] Result: {closestInfo} | Selected: {(best != null ? best.ClassName : "none")}");

            if (best == m_currentlySelected)
                return;

            if (m_currentlySelected != null)
                SetBoxColor(m_currentlySelected, m_defaultBoxColor);

            var previous = m_currentlySelected;
            m_currentlySelected = best;

            if (m_currentlySelected != null)
                SetBoxColor(m_currentlySelected, m_selectedBoxColor);

            if (best != previous)
                OnSelectionChanged?.Invoke(m_currentlySelected);
        }

        private void ClearSelection()
        {
            if (m_currentlySelected == null)
                return;
            SetBoxColor(m_currentlySelected, m_defaultBoxColor);
            m_currentlySelected = null;
            OnSelectionChanged?.Invoke(null);
        }

        private static void SetBoxColor(MultiDetectionUiManager.BoundingBoxData box, Color color)
        {
            if (box?.BoxRectTransform == null)
                return;
            var img = box.BoxRectTransform.GetComponent<Image>();
            if (img != null)
                img.color = color;
        }

        public MultiDetectionUiManager.BoundingBoxData CurrentSelection => m_currentlySelected;
    }
}
