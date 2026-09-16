using UnityEngine;

// Population creation moved to PythonBridge so every phenotype has a Python genome.
public class CreatureSpawner : MonoBehaviour
{
    public bool autoSpawnOnStart = false;
    public int creaturesToSpawn = 4;
    public float spawnSpacing = 5f;
}

public static class BodyUtils
{
    private static Sprite square;
    public static Sprite GetSquareSprite() {
        if (square != null) return square;
        Texture2D texture = new Texture2D(1, 1); texture.SetPixel(0, 0, Color.white); texture.Apply();
        square = Sprite.Create(texture, new Rect(0, 0, 1, 1), new Vector2(.5f,.5f), 1f); return square;
    }
    public static readonly Vector2[] SlotDirections = {
        Vector2.up, Vector2.down, Vector2.left, Vector2.right,
        new Vector2(1,1).normalized, new Vector2(1,-1).normalized,
        new Vector2(-1,1).normalized, new Vector2(-1,-1).normalized
    };
}

// Shared procedural-world constants. Keeping the spawn grid and environment
// extents in one place prevents respawned creatures, food, and predators from
// drifting back into a small centre-of-map cluster.
public static class WorldLayout
{
    // The normalization and physical boundary scale together. Checkpoints use
    // a matching configuration fingerprint, so older coordinate scales cannot
    // be resumed under this expanded arena.
    public const float WorldHalfWidth = 480f;
    public const float WorldBoundaryHalfWidth = 480f;
    public const float GoalBoundaryMargin = 8f;
    public const float GoalHalfWidth = WorldBoundaryHalfWidth - GoalBoundaryMargin;
    public const float GroundTop = -4.5f;
    public const int SpawnColumns = 5;
    public const int MaximumNativeSpawnSlots = 4;
    public const float SpawnColumnSpacing = 160f;
    public const float SpawnRowSpacing = 64f;

    public static Vector2 CreatureSpawnPosition(int slot)
    {
        int column = slot % SpawnColumns;
        int row = slot / SpawnColumns;
        return new Vector2((column-(SpawnColumns-1)*.5f)*SpawnColumnSpacing, GroundTop + 2.5f + row * SpawnRowSpacing);
    }

    public static bool IsFarFromCreatureSpawns(Vector2 point, float minimumDistance)
    {
        float minimumSqrDistance = minimumDistance * minimumDistance;
        for (int slot = 0; slot < MaximumNativeSpawnSlots; slot++)
            if ((point - CreatureSpawnPosition(slot)).sqrMagnitude < minimumSqrDistance)
                return false;
        return true;
    }
}
