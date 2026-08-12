using System.Collections.Generic;
using UnityEngine;

// Spawns ground + food + predators so Vision/touch/energy have something
// real to detect and interact with. Primitive-generated sprites, same
// approach as BodyUtils -- swap in real art/prefabs later without
// touching any detection code that reads these tags.
public class EnvironmentSpawner : MonoBehaviour
{
    [Header("Ground")]
    public float groundWidth = 100f;
    public float groundThickness = 2f;
    public Vector2 groundCenter = new Vector2(0f, -5f);

    [Header("Food")]
    public int foodCount = 10;
    public float foodSpawnRadius = 20f;
    public float foodEnergyValue = 20f;

    [Header("Predators")]
    public int predatorCount = 2;
    public float predatorSpawnRadius = 20f;

    private List<Food> spawnedFood = new List<Food>();

    private void Awake()
    {
        // fails loudly at scene start rather than the first time Vision's
        // CompareTag throws mid-simulation -- same class of problem as
        // the earlier deprecation/tag issues, caught before it can block
        // a whole session
        RequireTag("Food");
        RequireTag("Predator");
        RequireTag("Player");
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
        SpawnFood();
        SpawnPredators();
    }

    private void SpawnGround()
    {
        GameObject ground = new GameObject("Ground");
        ground.transform.position = groundCenter;

        SpriteRenderer sr = ground.AddComponent<SpriteRenderer>();
        sr.sprite = BodyUtils.GetSquareSprite();
        sr.color = new Color(0.4f, 0.3f, 0.2f);
        ground.transform.localScale = new Vector3(groundWidth, groundThickness, 1f);

        BoxCollider2D col = ground.AddComponent<BoxCollider2D>();
        col.size = Vector2.one;
        // no Rigidbody2D -- a Collider2D with no Rigidbody2D is a static
        // collider in Unity's 2D physics, exactly what ground should be
    }

    private void SpawnFood()
    {
        for (int i = 0; i < foodCount; i++)
        {
            Vector2 pos = groundCenter + Vector2.up * (groundThickness / 2f + 1f)
                          + Random.insideUnitCircle * foodSpawnRadius;

            GameObject go = new GameObject("Food");
            go.transform.position = pos;
            go.tag = "Food";

            SpriteRenderer sr = go.AddComponent<SpriteRenderer>();
            sr.sprite = BodyUtils.GetSquareSprite();
            sr.color = new Color(0.9f, 0.8f, 0.1f);
            go.transform.localScale = Vector3.one * 0.5f;

            CircleCollider2D col = go.AddComponent<CircleCollider2D>();
            col.radius = 0.5f;

            Food food = go.AddComponent<Food>();
            food.energyValue = foodEnergyValue;
            spawnedFood.Add(food);
        }
    }

    private void SpawnPredators()
    {
        for (int i = 0; i < predatorCount; i++)
        {
            Vector2 pos = groundCenter + Vector2.up * (groundThickness / 2f + 1f)
                          + Random.insideUnitCircle * predatorSpawnRadius;

            GameObject go = new GameObject("Predator");
            go.transform.position = pos;
            go.tag = "Predator";

            SpriteRenderer sr = go.AddComponent<SpriteRenderer>();
            sr.sprite = BodyUtils.GetSquareSprite();
            sr.color = new Color(0.7f, 0.1f, 0.1f);

            CircleCollider2D col = go.AddComponent<CircleCollider2D>();
            col.radius = 0.5f;
            col.isTrigger = true;

            go.AddComponent<Predator>();
        }
    }

    // Not wired to anything yet (item #10 in the inventory) -- kept ready
    // for whatever eventually manages energy/consumption cycles.
    public void RespawnFood(Food food)
    {
        food.consumed = false;
        food.gameObject.SetActive(true);
        food.transform.position = groundCenter + Vector2.up * (groundThickness / 2f + 1f)
                                   + Random.insideUnitCircle * foodSpawnRadius;
    }
}