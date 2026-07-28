using UnityEngine;

// A creature's persistent identity, independent of Unity's own internal
// object-identity system (which has been in flux -- see the GetInstanceID
// / GetEntityId deprecation). Every creature gets exactly one of these on
// its root GameObject; every body part (Torso, Limb) carries a reference
// to it via BodyPart, so touch/vision code can answer "is this the same
// creature," "same species," or neither, without touching Unity's own
// instance-id machinery at all.
public class CreatureIdentity : MonoBehaviour
{
    public string creatureId;

    // Placeholder until body speciation (see the python prototype's
    // body_compatibility_distance) is ported to C#. For now every spawned
    // creature gets the same speciesId from CreatureSpawner, so "same
    // species" checks are trivially true across the whole population --
    // real species grouping isn't wired up yet.
    public string speciesId;

    private void Awake()
    {
        // fallback only -- CreatureSpawner sets this explicitly right
        // after AddComponent, so this mostly matters if this component
        // ever gets added some other way
        if (string.IsNullOrEmpty(creatureId))
            creatureId = System.Guid.NewGuid().ToString();
    }
}