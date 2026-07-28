using System.Collections.Generic;
using UnityEngine;

public class CreatureSpawner : MonoBehaviour
{
    [Header("Spawn settings")]
    public int creaturesToSpawn = 1;
    public float spawnSpacing = 5f;

    void Start()
    {
        Debug.Log($"CreatureSpawner.Start() running. creaturesToSpawn = {creaturesToSpawn}");
        List<GameObject> result = SpawnCreatures(creaturesToSpawn, Vector2.zero, spawnSpacing);
        Debug.Log($"SpawnCreatures finished. {result.Count} creature(s) actually in the returned list.");
    }

    public List<GameObject> SpawnCreatures(int count, Vector2 origin, float spacing)
    {
        List<GameObject> spawned = new List<GameObject>();
        for (int i = 0; i < count; i++)
        {
            Vector2 spawnPos = origin + new Vector2(i * spacing, 0f);
            try
            {
                GameObject creature = SpawnCreature(spawnPos);
                spawned.Add(creature);

                // FIX: GetInstanceID() is deprecated (hard compile error
                // on recent Unity versions -- see explanation above). Log
                // the creature's own persistent creatureId instead, which
                // is more useful anyway since it survives independent of
                // Unity's own internal identity system.
                string shortId = creature.GetComponent<CreatureIdentity>().creatureId.Substring(0, 8);
                Debug.Log($"  Creature {i} spawned OK: {creature.name} (id {shortId})");
            }
            catch (System.Exception e)
            {
                Debug.LogError($"CreatureSpawner: creature {i} failed to spawn at {spawnPos} -- {e}");
            }
        }

        if (spawned.Count < count)
            Debug.LogWarning($"CreatureSpawner: requested {count}, only {spawned.Count} spawned successfully.");

        return spawned;
    }

    public GameObject SpawnCreature(Vector2 position)
    {
        GameObject root = new GameObject("Creature");
        root.transform.position = position;

        CreatureIdentity identity = root.AddComponent<CreatureIdentity>();
        identity.creatureId = System.Guid.NewGuid().ToString();
        identity.speciesId = "default_species"; // placeholder until body
                                                   // speciation is ported
                                                   // from the python side

        GameObject torsoGO = new GameObject("Torso");
        torsoGO.transform.SetParent(root.transform);
        torsoGO.transform.position = position;

        Torso torso = torsoGO.AddComponent<Torso>();
        torso.Init(identity);

        CreatureBrain brain = torsoGO.AddComponent<CreatureBrain>();
        brain.Init(torso, torso.GetAllLimbs());

        return root;
    }
}

public static class BodyUtils
{
    private static Sprite _squareSprite;

    public static Sprite GetSquareSprite()
    {
        if (_squareSprite != null) return _squareSprite;
        Texture2D tex = new Texture2D(1, 1);
        tex.SetPixel(0, 0, Color.white);
        tex.Apply();
        _squareSprite = Sprite.Create(tex, new Rect(0, 0, 1, 1), new Vector2(0.5f, 0.5f), 1f);
        return _squareSprite;
    }

    public static readonly Vector2[] SlotDirections =
    {
        Vector2.up, Vector2.down, Vector2.left, Vector2.right,
        new Vector2(1, 1).normalized, new Vector2(1, -1).normalized,
        new Vector2(-1, 1).normalized, new Vector2(-1, -1).normalized
    };
}