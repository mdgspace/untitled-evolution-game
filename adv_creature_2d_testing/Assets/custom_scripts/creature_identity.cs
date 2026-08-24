using UnityEngine;

public class CreatureIdentity : MonoBehaviour
{
    public string creatureId;
    public string speciesId;
    public string bodyGenomeJson;

    // Set once by CreatureSpawner right after Torso.Init() -- lets Food
    // (and later, Predator damage, etc) reach a creature's energy without
    // a hierarchy search on every collision.
    public Torso torso;

    private void Awake()
    {
        if (string.IsNullOrEmpty(creatureId))
            creatureId = System.Guid.NewGuid().ToString();
    }
}
