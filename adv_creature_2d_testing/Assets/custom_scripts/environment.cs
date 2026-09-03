using System.Collections.Generic;
using UnityEngine;

// Spawns ground + food + predators so Vision/touch/energy have something
// real to detect and interact with. Primitive-generated sprites, same
// approach as BodyUtils -- swap in real art/prefabs later without
// touching any detection code that reads these tags.
public class EnvironmentSpawner : MonoBehaviour
{
    [Header("Ground")]
    public float groundWidth = WorldLayout.WorldBoundaryHalfWidth * 2f;
    public float groundThickness = 1f;
    public Vector2 groundCenter = new Vector2(0f, -5f);

    [Header("Food")]
    public int foodCount = 10;
    public float foodSpawnRadius = 8f;
    public float foodEnergyValue = 20f;
    public float foodScale = .22f;
    public float foodMinimumSpawnDistance = 1.8f;

    [Header("Native training routine")]
    public bool trainingRoutineEnabled = true;
    public float trainingEpisodeSeconds = 10f;
    public int trainingFoodPerCreature = 3;
    public float trainingFoodMinimumDistance = 5f;
    public float trainingFoodMaximumDistance = 14f;
    public float highEnergyFoodValue = 150f;
    public float highEnergyFoodSpeed = 3.5f;
    [Tooltip("Food and predators are created only after the native feature button is pressed.")]
    public bool startFeaturesEnabled = false;

    [Header("Predators")]
    // Keep a single predator in the native training arena. Predators remain
    // present as a safety/avoidance signal, but do not dominate early body
    // learning or remove most candidates before M2 has a useful policy.
    public int predatorCount = 1;
    public float predatorSpawnRadius = 120f;
    public float predatorScale = .35f;
    public float predatorMinimumSpawnDistance = 28f;
    public float trainingPredatorMinimumDistance = 24f;
    public float trainingPredatorMaximumDistance = 34f;
    public float predatorChaseSpeed = .25f;

    private List<Food> spawnedFood = new List<Food>();
    private List<HighEnergyFood> spawnedHighEnergyFood = new List<HighEnergyFood>();
    private List<Predator> spawnedPredators = new List<Predator>();
    private System.Random trainingRandom = new System.Random(7301);
    private float nextTrainingEpisode;
    public bool FeaturesEnabled { get; private set; }

    private void Awake()
    {
        // fails loudly at scene start rather than the first time Vision's
        // CompareTag throws mid-simulation -- same class of problem as
        // the earlier deprecation/tag issues, caught before it can block
        // a whole session
        RequireTag("Food");
        RequireTag("Predator");
    }

    private void RequireTag(string tag)
    {
        try
        {
            GameObject probe = new GameObject("TagProbe");
            probe.tag = tag;
            Destroy(probe);
        }
        catch (UnityException)
        {
            Debug.LogError($"EnvironmentSpawner: tag '{tag}' isn't registered. " +
                            "Add it in Project Settings -> Tags and Layers before running.");
        }
    }

    private void Start()
    {
        SpawnGround();
        if (startFeaturesEnabled) EnableFeatures();
    }

    private void FixedUpdate()
    {
        if (!FeaturesEnabled || !trainingRoutineEnabled || Time.time < nextTrainingEpisode) return;
        nextTrainingEpisode += Mathf.Max(.5f, trainingEpisodeSeconds);
        RunTrainingEpisode();
    }

    public void EnableFeatures()
    {
        if (FeaturesEnabled) return;
        FeaturesEnabled = true;
        SpawnFood();
        SpawnPredators();
        EnsureTrainingPools();
        nextTrainingEpisode = Time.time + trainingEpisodeSeconds;
    }

    // NativeEcosystemController calls this during startup so Inspector values
    // cannot bypass the explicit telemetry-button activation path.
    public void LockFeaturesUntilExplicitEnable()
    {
        startFeaturesEnabled = false;
        DisableFeatures();
    }

    public void DisableFeatures()
    {
        FeaturesEnabled = false;
        foreach (Food food in spawnedFood) if (food != null) Destroy(food.gameObject);
        foreach (HighEnergyFood food in spawnedHighEnergyFood) if (food != null) Destroy(food.gameObject);
        foreach (Predator predator in spawnedPredators) if (predator != null) Destroy(predator.gameObject);
        spawnedFood.Clear(); spawnedHighEnergyFood.Clear(); spawnedPredators.Clear();
    }

    private void SpawnGround()
    {
        if (GameObject.Find("Ground") != null) return;
        GameObject ground = new GameObject("Ground");
        ground.transform.position = groundCenter;

        SpriteRenderer sr = ground.AddComponent<SpriteRenderer>();
        sr.sprite = BodyUtils.GetSquareSprite();
        sr.color = new Color(0.4f, 0.3f, 0.2f);
        ground.transform.localScale = new Vector3(groundWidth, groundThickness, 1f);

        BoxCollider2D col = ground.AddComponent<BoxCollider2D>();
        col.size = Vector2.one; col.sharedMaterial = NativePhysicsMaterials.Ground; ground.AddComponent<GroundSurface>();
        // no Rigidbody2D -- a Collider2D with no Rigidbody2D is a static
        // collider in Unity's 2D physics, exactly what ground should be
        SpawnWall("WorldWallLeft", groundCenter.x - groundWidth / 2f);
        SpawnWall("WorldWallRight", groundCenter.x + groundWidth / 2f);
    }

    private void SpawnWall(string name, float x)
    {
        if (GameObject.Find(name) != null) return;
        GameObject wall = new GameObject(name); wall.transform.position = new Vector2(x, groundCenter.y + 20f);
        BoxCollider2D collider = wall.AddComponent<BoxCollider2D>(); collider.size = new Vector2(1f, 50f);
    }

    private void SpawnFood()
    {
        for (int i = 0; i < foodCount; i++)
        {
            Vector2 pos = InitialFoodPosition(i);
            spawnedFood.Add(CreateFood(pos));
        }
    }

    private Food CreateFood(Vector2 position)
    {
        GameObject go = new GameObject("Food"); go.transform.position = position; go.tag = "Food";
        SpriteRenderer sr = go.AddComponent<SpriteRenderer>(); sr.sprite = BodyUtils.GetSquareSprite(); sr.color = new Color(0.9f, 0.8f, 0.1f); go.transform.localScale = Vector3.one * foodScale;
        CircleCollider2D col = go.AddComponent<CircleCollider2D>(); col.radius = 0.5f; col.isTrigger = true;
        Food food = go.AddComponent<Food>(); food.energyValue = foodEnergyValue; food.respawnDelay = 5f; food.Init(this); return food;
    }

    private void SpawnPredators()
    {
        for (int i = 0; i < predatorCount; i++)
        {
            Vector2 pos = InitialPredatorPosition(i);

            GameObject go = new GameObject("Predator");
            go.transform.position = pos;
            go.tag = "Predator";

            SpriteRenderer sr = go.AddComponent<SpriteRenderer>();
            sr.sprite = BodyUtils.GetSquareSprite();
            sr.color = new Color(0.7f, 0.1f, 0.1f);
            go.transform.localScale = Vector3.one * predatorScale;

            CircleCollider2D col = go.AddComponent<CircleCollider2D>();
            col.radius = 0.5f;
            col.isTrigger = true;

            Predator predator = go.AddComponent<Predator>(); predator.chaseSpeed = predatorChaseSpeed; spawnedPredators.Add(predator);
        }
    }

    private void EnsureTrainingPools()
    {
        int creatureCount = Mathf.Max(1, FindObjectsByType<CreatureIdentity>(FindObjectsInactive.Exclude).Length);
        int desiredFood = creatureCount * Mathf.Max(1, trainingFoodPerCreature);
        while (spawnedFood.Count < desiredFood) spawnedFood.Add(CreateFood(InitialFoodPosition(spawnedFood.Count)));
        while (spawnedHighEnergyFood.Count < creatureCount)
        {
            GameObject go = new GameObject("HighEnergyFood"); go.tag = "Food";
            go.transform.localScale = Vector3.one * (foodScale * 1.35f);
            SpriteRenderer sr = go.AddComponent<SpriteRenderer>(); sr.sprite = BodyUtils.GetSquareSprite(); sr.color = new Color(0.15f, 0.95f, 0.95f);
            CircleCollider2D col = go.AddComponent<CircleCollider2D>(); col.radius = 0.5f; col.isTrigger = true;
            HighEnergyFood food = go.AddComponent<HighEnergyFood>(); food.energyValue = highEnergyFoodValue; food.escapeSpeed = highEnergyFoodSpeed; food.surfaceY = groundCenter.y + groundThickness / 2f + foodScale * .5f + .05f; food.PlaceAt(InitialFoodPosition(spawnedHighEnergyFood.Count)); spawnedHighEnergyFood.Add(food);
        }
    }

    private void RunTrainingEpisode()
    {
        CreatureIdentity[] living = FindObjectsByType<CreatureIdentity>(FindObjectsInactive.Exclude);
        if (living.Length == 0) return;
        EnsureTrainingPools();
        float foodHeight = groundCenter.y + groundThickness / 2f + foodScale * .5f + .05f;
        for (int i = 0; i < spawnedFood.Count; i++)
        {
            CreatureIdentity target = living[i % living.Length];
            float distance = Range(trainingFoodMinimumDistance, trainingFoodMaximumDistance);
            float side = trainingRandom.Next(2) == 0 ? -1f : 1f;
            Vector2 point = new Vector2(target.torso.transform.position.x + side * distance, foodHeight);
            point.x = Mathf.Clamp(point.x, -WorldLayout.WorldBoundaryHalfWidth + 2f, WorldLayout.WorldBoundaryHalfWidth - 2f);
            spawnedFood[i].PlaceAt(point);
        }
        for (int i = 0; i < spawnedHighEnergyFood.Count; i++)
        {
            CreatureIdentity target = living[i % living.Length];
            float distance = Range(trainingFoodMaximumDistance, trainingFoodMaximumDistance + 8f);
            float side = trainingRandom.Next(2) == 0 ? -1f : 1f;
            Vector2 point = new Vector2(target.torso.transform.position.x + side * distance, foodHeight);
            point.x = Mathf.Clamp(point.x, -WorldLayout.WorldBoundaryHalfWidth + 2f, WorldLayout.WorldBoundaryHalfWidth - 2f);
            spawnedHighEnergyFood[i].PlaceAt(point);
        }
        for (int i = 0; i < spawnedPredators.Count; i++)
        {
            CreatureIdentity target = living[(i + 1) % living.Length];
            float distance = Range(trainingPredatorMinimumDistance, trainingPredatorMaximumDistance);
            float side = trainingRandom.Next(2) == 0 ? -1f : 1f;
            Vector2 point = new Vector2(target.torso.transform.position.x + side * distance, foodHeight + .15f);
            point.x = Mathf.Clamp(point.x, -WorldLayout.WorldBoundaryHalfWidth + 3f, WorldLayout.WorldBoundaryHalfWidth - 3f);
            spawnedPredators[i].PlaceAt(point);
        }
    }

    private float Range(float minimum, float maximum) => minimum + (float)trainingRandom.NextDouble() * (maximum - minimum);

    public void RespawnFood(Food food)
    {
        food.consumed = false;
        food.GetComponent<Collider2D>().enabled = true;
        food.GetComponent<SpriteRenderer>().enabled = true;
        food.transform.position = RandomFoodPosition();
    }

    private Vector2 InitialFoodPosition(int index)
    {
        // One visible, reachable food item per starting spawn.  It is far
        // enough to require locomotion, but inside the eight-unit M1 vision
        // radius from the first control observation.
        Vector2 spawn = WorldLayout.CreatureSpawnPosition(index % WorldLayout.MaximumNativeSpawnSlots);
        float side = index % 2 == 0 ? 1f : -1f;
        return new Vector2(spawn.x + side * foodMinimumSpawnDistance,
            groundCenter.y + groundThickness / 2f + foodScale * .5f + .05f);
    }

    private Vector2 InitialPredatorPosition(int index)
    {
        int slot = (index * 3 + 1) % WorldLayout.MaximumNativeSpawnSlots;
        Vector2 spawn = WorldLayout.CreatureSpawnPosition(slot);
        float height = groundCenter.y + groundThickness / 2f + predatorScale * .5f + .05f;
        return new Vector2(spawn.x + (index % 2 == 0 ? predatorMinimumSpawnDistance : -predatorMinimumSpawnDistance), height);
    }

    private Vector2 RandomFoodPosition()
    {
        CreatureIdentity[] living = FindObjectsByType<CreatureIdentity>(FindObjectsInactive.Exclude);
        float height = groundCenter.y + groundThickness / 2f + foodScale * .5f + .05f;
        for (int attempt = 0; attempt < 32 && living.Length > 0; attempt++)
        {
            CreatureIdentity target = living[Random.Range(0, living.Length)];
            if (target == null || target.torso == null) continue;
            // Keep the next reward in the creature's vision range.  Food is
            // not a free pickup: the minimum still requires real translation.
            float distance = Random.Range(foodMinimumSpawnDistance,
                foodMinimumSpawnDistance + Mathf.Min(1.2f, foodSpawnRadius));
            float direction = Random.value < .5f ? -1f : 1f;
            Vector2 candidate = new Vector2(target.torso.transform.position.x + direction * distance, height);
            if (Mathf.Abs(candidate.x) < WorldLayout.WorldBoundaryHalfWidth - 1f &&
                IsClearOfLivingCreatures(candidate, foodScale + .15f)) return candidate;
        }
        return InitialFoodPosition(Random.Range(0, foodCount));
    }

    private Vector2 RandomDistantPosition(float radius, float minimumDistance, float heightAboveGround)
    {
        float height = groundCenter.y + groundThickness / 2f + heightAboveGround;
        for (int attempt = 0; attempt < 32; attempt++) {
            Vector2 point = new Vector2(Random.Range(-radius, radius), height);
            if (WorldLayout.IsFarFromCreatureSpawns(point, minimumDistance) && IsClearOfLivingCreatures(point, minimumDistance)) return point;
        }
        // A deterministic edge fallback prevents an infinite spawn loop in a
        // crowded future map configuration.
        return new Vector2(-radius, height);
    }

    private bool IsClearOfLivingCreatures(Vector2 point, float minimumDistance)
    {
        float squared = minimumDistance * minimumDistance;
        foreach (CreatureIdentity identity in FindObjectsByType<CreatureIdentity>(FindObjectsInactive.Exclude))
            if (identity != null && identity.torso != null && ((Vector2)identity.torso.transform.position - point).sqrMagnitude < squared)
                return false;
        return Physics2D.OverlapCircle(point, Mathf.Max(.25f, foodScale)) == null;
    }
}
