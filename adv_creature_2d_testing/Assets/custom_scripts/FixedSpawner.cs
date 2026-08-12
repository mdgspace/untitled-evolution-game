using System.Collections.Generic;
using UnityEngine;

public class FixedSpawner : MonoBehaviour
{
    [Header("Spawn Settings")]
    public Vector2 spawnPosition = Vector2.zero;
    public string fixedCreatureId = "fixed_creature_0";

    void Start()
    {
        Debug.Log($"FixedSpawner: Spawning deterministic creature '{fixedCreatureId}' at {spawnPosition}");
        SpawnFixedCreature(spawnPosition);
    }

    public GameObject SpawnFixedCreature(Vector2 position)
    {
        GameObject root = new GameObject("FixedCreature");
        root.transform.position = position;

        CreatureIdentity identity = root.AddComponent<CreatureIdentity>();
        identity.creatureId = fixedCreatureId;
        identity.speciesId = "fixed_species";

        GameObject torsoGO = new GameObject("Torso");
        torsoGO.transform.SetParent(root.transform);
        torsoGO.transform.position = position;

        Torso torso = torsoGO.AddComponent<Torso>();
        
        // Initialize torso with fixed dimensions (1.5 x 1.5)
        torso.dimensions = new Vector2(1.5f, 1.5f);
        torso.Init(identity);
        identity.torso = torso;

        CreatureBrain brain = torsoGO.AddComponent<CreatureBrain>();
        brain.Init(torso, torso.GetAllLimbs());

        Debug.Log($"FixedSpawner: Successfully spawned fixed creature with {torso.GetAllLimbs().Count} limbs.");
        return root;
    }
}
