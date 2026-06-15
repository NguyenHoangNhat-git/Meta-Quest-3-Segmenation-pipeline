// ObjectInfoDatabase.cs
using System;
using System.Collections.Generic;
using Newtonsoft.Json; // com.unity.nuget.newtonsoft-json
using Newtonsoft.Json.Linq;
using UnityEngine;

namespace CVMQ3.ProcessingPipeline
{
    public class ObjectInfoDatabase
    {
        private readonly Dictionary<int, ObjectInfoEntry> _byId;

        private ObjectInfoDatabase(Dictionary<int, ObjectInfoEntry> byId)
        {
            _byId = byId;
        }

        public static ObjectInfoDatabase Load(TextAsset jsonAsset)
        {
            if (jsonAsset == null)
            {
                Debug.LogError("[ObjectInfoDatabase] TextAsset is null.");
                return null;
            }

            try
            {
                // Parse the root object so we can grab just the "classes" array,
                // ignoring "metadata" and any other top-level keys entirely.
                var root = JObject.Parse(jsonAsset.text.Replace("\r\n", "\n").Replace("\r", "\n"));
                var classesToken = root["classes"];

                if (classesToken == null)
                {
                    Debug.LogError(
                        "[ObjectInfoDatabase] JSON has no \"classes\" key at root. "
                            + $"Found keys: {string.Join(", ", root.Properties())}"
                    );
                    return null;
                }

                var entries = classesToken.ToObject<List<ObjectInfoEntry>>();

                if (entries == null || entries.Count == 0)
                {
                    Debug.LogError("[ObjectInfoDatabase] \"classes\" array parsed but is empty.");
                    return null;
                }

                var dict = new Dictionary<int, ObjectInfoEntry>(entries.Count);
                foreach (var entry in entries)
                {
                    if (entry == null)
                        continue;
                    if (dict.ContainsKey(entry.id))
                        Debug.LogWarning(
                            $"[ObjectInfoDatabase] Duplicate id {entry.id} — keeping first."
                        );
                    else
                    {
                        dict[entry.id] = entry;
                        Debug.Log($"[ObjectInfoDatabase] Entry id: {entry.id} | Entry");
                        Debug.Log($"[ObjectInfoDatabase] - {entry.name}");
                        Debug.Log($"[ObjectInfoDatabase] - {entry.aliases}");
                        Debug.Log($"[ObjectInfoDatabase] - {entry.category}");
                        Debug.Log($"[ObjectInfoDatabase] - {entry.pose_estimation_notes}");
                    }
                }

                Debug.Log($"[ObjectInfoDatabase] Loaded {dict.Count} entries.");
                return new ObjectInfoDatabase(dict);
            }
            catch (Exception ex)
            {
                Debug.LogError($"[ObjectInfoDatabase] JSON parse failed: {ex.Message}");
                return null;
            }
        }

        public bool TryGet(int classId, out ObjectInfoEntry entry) =>
            _byId.TryGetValue(classId, out entry);
    }
}
