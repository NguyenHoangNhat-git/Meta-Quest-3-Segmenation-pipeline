using System.Collections.Generic;
using Meta.XR;
using Meta.XR.Samples;
using UnityEngine;
using UnityEngine.Events;
using UnityEngine.UI;

namespace CVMQ3.ProcessingPipeline
{
    /// <summary>
    /// Receives detection results and projects each bounding box into world space
    /// using depth raycasting. Manages a prefab pool to avoid per-frame instantiation.
    /// </summary>
    [MetaCodeSample("PassthroughCameraApiSamples-MultiObjectDetection")]
    public class MultiDetectionUiManager : MonoBehaviour
    {
        [Header("Placement configuration")]
        [SerializeField]
        private EnvironmentRayCastSampleManager m_environmentRaycast;

        [SerializeField]
        private PassthroughCameraAccess m_cameraAccess;

        [SerializeField]
        private RectTransform m_detectionBoxPrefab;

        [Space(10)]
        public UnityEvent<int> OnObjectsDetected;

        [Header("Object Info JSON")]
        [SerializeField]
        private TextAsset m_objectInfoJson;

        // Active boxes currently visible in the scene.
        internal readonly List<BoundingBoxData> m_boxDrawn = new();
        private string[] m_labels;

        // Pool of inactive box RectTransforms to reuse instead of instantiating.
        private readonly List<BoundingBoxData> m_boxPool = new();

        // Seconds a box persists without a matching detection before being pooled.
        private float timeToPersistBoxes = 0.75f;

        public class BoundingBoxData
        {
            public string ClassName;
            public int ClassId;
            public RectTransform BoxRectTransform;
            public float lastUpdateTime;
            public Vector4 NormalizedBBox; // x1,y1,x2,y2 normalized 0-1
        }

        private void Awake()
        {
            Debug.Log("[Diag] Inference Manager UI started");
            m_detectionBoxPrefab.gameObject.SetActive(false);
        }

        private void Update()
        {
            // Remove boxes that haven't been updated recently
            for (int i = m_boxDrawn.Count - 1; i >= 0; i--)
            {
                var box = m_boxDrawn[i];
                if (Time.time - box.lastUpdateTime > timeToPersistBoxes)
                {
                    ReturnToPool(box);
                    m_boxDrawn.RemoveAt(i);
                }
            }
        }

        public void SetLabels(TextAsset labelsAsset)
        {
            // Parse neural net labels
            m_labels = labelsAsset.text.Split('\n');
        }

        /// <summary>
        /// Projects each detection into world space and updates or creates bounding box UI elements.
        /// Uses depth raycasting to place boxes on real surfaces at the correct depth.
        /// </summary>
        public void DrawUIBoxes(
            List<(int classId, Vector4 boundingBox)> detections,
            Vector2 inputSize,
            Pose cameraPose
        )
        {
            Vector2 currentResolution = m_cameraAccess.CurrentResolution;

            if (detections.Count == 0)
            {
                OnObjectsDetected?.Invoke(0);
                return;
            }
            Debug.Log("[Diag] UI receives detections");
            OnObjectsDetected?.Invoke(detections.Count);

            // Draw the bounding boxes
            for (var i = 0; i < detections.Count; i++)
            {
                var detection = detections[i];
                float x1 = detection.boundingBox[0];
                float y1 = detection.boundingBox[1];
                float x2 = detection.boundingBox[2];
                float y2 = detection.boundingBox[3];
                Rect rect = new Rect(x1, y1, x2 - x1, y2 - y1);
                // Rect rect = Rect.MinMaxRect(x1, y1, x2, y2); // todo

                Vector2 normalizedCenter = rect.center / inputSize;
                Vector2 center = currentResolution * (normalizedCenter - Vector2.one * 0.5f);

                var classname = m_labels[detection.classId].Replace(" ", "_");

                // Cast a ray from the camera through the box centre to find the world surface.
                var ray = m_cameraAccess.ViewportPointToRay(
                    new Vector2(normalizedCenter.x, 1.0f - normalizedCenter.y),
                    cameraPose
                );
                var worldPos = m_environmentRaycast.Raycast(ray);
                if (!worldPos.HasValue)
                {
                    Debug.Log($"RaycastManager failed, ray:{ray}, cameraPose:{cameraPose}");
                    continue;
                }

                // Normalised rect in Y-up viewport space for downstream use (crop extractor).
                var normRect = new Rect(
                    rect.x / inputSize.x,
                    1f - rect.yMax / inputSize.y,
                    rect.width / inputSize.x,
                    rect.height / inputSize.y
                );

                // Calculate distance and center point first
                float distance = Vector3.Distance(cameraPose.position, worldPos.Value);
                var worldSpaceCenter = m_cameraAccess
                    .ViewportPointToRay(normRect.center, cameraPose)
                    .GetPoint(distance);
                var normal = (worldSpaceCenter - cameraPose.position).normalized;

                // Project bbox corners onto the plane perpendicular to the view ray at the hit point,
                // then measure the world-space extent to size the RectTransform correctly.
                var plane = new Plane(normal, worldSpaceCenter);
                var minRay = m_cameraAccess.ViewportPointToRay(normRect.min, cameraPose);
                var maxRay = m_cameraAccess.ViewportPointToRay(normRect.max, cameraPose);
                plane.Raycast(minRay, out float intersectionDistanceMin);
                plane.Raycast(maxRay, out float intersectionDistanceMax);
                var min = minRay.GetPoint(intersectionDistanceMin);
                var max = maxRay.GetPoint(intersectionDistanceMax);

                // Transform to camera-local space to get a 2D size free of perspective distortion.
                var topLeftLocal =
                    Quaternion.Inverse(cameraPose.rotation) * (min - cameraPose.position);
                var bottomRightLocal =
                    Quaternion.Inverse(cameraPose.rotation) * (max - cameraPose.position);
                var size = new Vector2(
                    Mathf.Abs(bottomRightLocal.x - topLeftLocal.x),
                    Mathf.Abs(bottomRightLocal.y - topLeftLocal.y)
                );

                var boxData = GetOrCreateBoundingBoxData(detection.classId, worldSpaceCenter, size);

                // Store normalized coords directly from the detection
                boxData.NormalizedBBox = new Vector4(
                    x1 / inputSize.x,
                    y1 / inputSize.y,
                    x2 / inputSize.x,
                    y2 / inputSize.y
                );
                Debug.Log("[Diag] UI got bounding box data");
                var boxRectTransform = boxData.BoxRectTransform;
                boxRectTransform.GetComponentInChildren<Text>().text =
                    $"Id: {detection.classId} Class: {classname} Center (px): {center:0.0} Center (%): {normalizedCenter:0.0}";
                boxRectTransform.SetPositionAndRotation(
                    worldSpaceCenter,
                    Quaternion.LookRotation(normal)
                );
                boxRectTransform.sizeDelta = size;
                boxData.lastUpdateTime = Time.time;
            }
            Debug.Log("[Diag] Finished drawing UI");
        }

        /// <summary>
        /// Returns an existing box that overlaps the new detection (same class) or creates a new one.
        /// Removes overlapping boxes of a different class to avoid UI clutter.
        /// </summary>
        private BoundingBoxData GetOrCreateBoundingBoxData(
            int classId,
            Vector3 worldSpaceCenter,
            Vector2 worldSpaceSize
        )
        {
            BoundingBoxData reusedBox = null;
            for (int i = m_boxDrawn.Count - 1; i >= 0; i--)
            {
                var box = m_boxDrawn[i];
                var localPos = box.BoxRectTransform.InverseTransformPoint(worldSpaceCenter);
                var newBox = new Vector4(
                    localPos.x - worldSpaceSize.x * 0.5f,
                    localPos.y - worldSpaceSize.y * 0.5f,
                    localPos.x + worldSpaceSize.x * 0.5f,
                    localPos.y + worldSpaceSize.y * 0.5f
                );

                var sizeDelta = box.BoxRectTransform.sizeDelta;
                var currentBox = new Vector4(
                    -sizeDelta.x * 0.5f,
                    -sizeDelta.y * 0.5f,
                    sizeDelta.x * 0.5f,
                    sizeDelta.y * 0.5f
                );

                if (box.ClassId == classId)
                {
                    // If the new box overlaps with an existing one of the same class, reuse it
                    if (MultiDetectionRunManager.CalculateIoU(newBox, currentBox) > 0f)
                    {
                        if (reusedBox == null)
                        {
                            reusedBox = box;
                        }
                        else
                        {
                            // Same overlapping class - remove the existing box
                            ReturnToPool(box);
                            m_boxDrawn.RemoveAt(i);
                        }
                    }
                }
                // If the new box's IoU with another class is significant, remove the existing box
                else if (MultiDetectionRunManager.CalculateIoU(newBox, currentBox) > 0.1f)
                {
                    // Different overlapping class - remove the existing box
                    ReturnToPool(box);
                    m_boxDrawn.RemoveAt(i);
                }
            }

            if (reusedBox != null)
            {
                return reusedBox;
            }

            // Create a new box
            var newData = GetBoxFromPoolOrCreate();
            newData.ClassId = classId;
            newData.ClassName = m_labels[classId].Replace(" ", "_");
            m_boxDrawn.Add(newData);
            return newData;
        }

        private BoundingBoxData GetBoxFromPoolOrCreate()
        {
            if (m_boxPool.Count > 0)
            {
                var pooled = m_boxPool[m_boxPool.Count - 1];
                pooled.BoxRectTransform.gameObject.SetActive(true);
                m_boxPool.RemoveAt(m_boxPool.Count - 1);
                return pooled;
            }

            var boxRectTransform = Instantiate(m_detectionBoxPrefab, ContentParent);
            boxRectTransform.gameObject.SetActive(true);
            return new BoundingBoxData { BoxRectTransform = boxRectTransform };
        }

        internal Transform ContentParent => m_detectionBoxPrefab.parent;

        private void ReturnToPool(BoundingBoxData box)
        {
            box.BoxRectTransform.gameObject.SetActive(false);
            m_boxPool.Add(box);
        }

        internal void ClearAnnotations()
        {
            foreach (var box in m_boxDrawn)
            {
                ReturnToPool(box);
            }
            m_boxDrawn.Clear();
        }
    }
}
