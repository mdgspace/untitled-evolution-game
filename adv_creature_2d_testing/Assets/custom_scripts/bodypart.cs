using UnityEngine;

// Attached automatically to every Torso and Limb GameObject at spawn time
// (see Torso.Init / Limb.Init). Lets collision/vision code identify which
// creature -- and which species -- a given collider belongs to, via
// GetComponent<BodyPart>() on whatever was hit.
public class BodyPart : MonoBehaviour
{
    public CreatureIdentity identity;
}