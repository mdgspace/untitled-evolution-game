using System.Collections;
using System.Collections.Generic;
using Stopwatch = System.Diagnostics.Stopwatch;
using System.Linq;
using System.Reflection;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.TestTools;

public class LiveEcosystemPlayModeTests
{
    [UnityTearDown]
    public IEnumerator RestoreClock()
    {
        Time.timeScale = 1f;
        yield return null;
    }

    [UnityTest]
    public IEnumerator SampleSceneBootstrapsNativeEcosystemWithoutPython()
    {
        SceneManager.LoadScene("SampleScene", LoadSceneMode.Single);
        yield return null;
        float deadline = Time.realtimeSinceStartup + 10f;
        NativeEcosystemController native = null;
        while (Time.realtimeSinceStartup < deadline) {
            native = Object.FindAnyObjectByType<NativeEcosystemController>();
            if (native != null && native.Population == native.populationTarget) break;
            yield return null;
        }
        Assert.NotNull(native);
        Assert.AreEqual(.02f,Time.fixedDeltaTime,1e-7f);
        Assert.AreEqual(50f,native.PhysicsRateHz,1e-4f);
        Assert.AreEqual(0, Object.FindObjectsByType<PythonBridge>(FindObjectsInactive.Exclude).Count(component => component.isActiveAndEnabled));
        Assert.AreEqual(75f,native.evolutionIntervalSeconds);
        Assert.AreEqual(12,native.bodyMutationCooldownGenerations);
        EnvironmentSpawner environment = Object.FindAnyObjectByType<EnvironmentSpawner>();
        Assert.NotNull(environment);
        Assert.AreEqual(1, Object.FindObjectsByType<EnvironmentSpawner>(FindObjectsInactive.Exclude).Length);
        Assert.AreEqual(10, native.Population, native.LastCheckpointError);
        Assert.That(native.SpeciesCount,Is.InRange(1,NativeEcosystemController.InitialBodySpeciesCount),native.LoadedFromCheckpoint?"Loaded populations retain their persisted species":"Fresh bodies are grouped by the morphology distance threshold");
        Assert.NotNull(Object.FindAnyObjectByType<ObserverCamera>());
        Assert.NotNull(GameObject.Find("WorldWallLeft"));
        Assert.NotNull(GameObject.Find("WorldWallRight"));
        Food[] food = Object.FindObjectsByType<Food>(FindObjectsInactive.Include);
        Predator[] predators = Object.FindObjectsByType<Predator>(FindObjectsInactive.Include);
        Assert.IsFalse(native.M1Enabled);Assert.AreEqual("locked at startup",native.M1ActivationSource);Assert.IsFalse(native.PopulationDynamicsEnabled);Assert.IsFalse(environment.FeaturesEnabled);Assert.AreEqual(0,food.Length);Assert.AreEqual(0,predators.Length);
        yield return new WaitForFixedUpdate();Assert.IsFalse(native.M1Enabled);Assert.IsFalse(native.PopulationDynamicsEnabled);Assert.AreEqual("locked at startup",native.M1ActivationSource);
    }

    [UnityTest]
    [Timeout(240000)]
    public IEnumerator OpenLoopControllerSurvivesScaledBiasDecayAndReplacement()
    {
        SceneManager.LoadScene("SampleScene", LoadSceneMode.Single);
        yield return null;
        NativeEcosystemController native = null;
        float setupDeadline = Time.realtimeSinceStartup + 15f;
        while (Time.realtimeSinceStartup < setupDeadline) {
            native = Object.FindAnyObjectByType<NativeEcosystemController>();
            if (native != null && native.Population == native.populationTarget) break;
            yield return null;
        }
        Assert.NotNull(native);
        Assert.AreEqual(native.populationTarget, native.Population, native.LastCheckpointError);

        CreatureBrain protectedCreature=Object.FindObjectsByType<CreatureBrain>(FindObjectsInactive.Exclude).FirstOrDefault(b=>!b.IsDead);Assert.NotNull(protectedCreature);protectedCreature.Kill("pre-activation death must be blocked");Assert.IsFalse(protectedCreature.IsDead);
        native.EnableInteractiveFeaturesFromUser();Assert.IsTrue(native.M1Enabled);Assert.IsTrue(native.PopulationDynamicsEnabled);
        int startTick = native.Tick;
        int startM3 = native.SharedM3Updates;
        int startReplacements = native.ImmediateReplacementCount;
        bool forcedDeath = false;
        var actionMagnitudes = new System.Collections.Generic.HashSet<int>();
        Time.timeScale = 100f;
        float deadline = Time.realtimeSinceStartup + 210f;
        while (native != null && native.Tick - startTick < NativeCreatureModel.ActionBiasEndTick && Time.realtimeSinceStartup < deadline) {
            if (native.LastActionMagnitude > 0f) actionMagnitudes.Add(Mathf.RoundToInt(native.LastActionMagnitude * 1000f));
            if (!forcedDeath && native.Tick - startTick >= 90) {
                CreatureBrain victim = Object.FindObjectsByType<CreatureBrain>(FindObjectsInactive.Exclude).FirstOrDefault(b => !b.IsDead);
                Assert.NotNull(victim);
                victim.Kill("open-loop replacement acceptance test");
                forcedDeath = true;
            }
            yield return null;
        }

        Assert.NotNull(native);
        Assert.GreaterOrEqual(native.Tick - startTick, NativeCreatureModel.ActionBiasEndTick, "Scaled bias-decay physics ticks did not finish before the real-time deadline");
        Assert.AreEqual(native.populationTarget, native.Population, native.ReplacementRetryState);
        Assert.Greater(native.ImmediateReplacementCount, startReplacements, "Forced death was not replaced");
        Assert.Greater(native.SharedM3Updates, startM3 + 100, "Shared M3 did not continue training");
        Assert.Greater(native.LastM3BatchSize,0,"No completed creature sequence reached the shared M3 batch");
        Assert.Greater(actionMagnitudes.Count, 2, "Executed action magnitudes lacked diversity");
        Assert.GreaterOrEqual(native.BiasStrength,.5f,"Unproven controllers must retain excitation after the nominal bias cutoff");
        Assert.AreEqual(0, native.ControlFailures, native.LastControlError);
        Assert.IsFalse(float.IsNaN(native.LastM2Loss) || float.IsInfinity(native.LastM2Loss));
        Assert.IsFalse(float.IsNaN(native.LastM3Loss) || float.IsInfinity(native.LastM3Loss));
        Assert.That(native.SequenceStep, Is.InRange(0, NativeCreatureModel.PlanningHorizon));
        Debug.Log($"OPEN_LOOP_SCALED tick_delta={native.Tick-startTick} population={native.Population} replacements={native.ImmediateReplacementCount-startReplacements} m3_updates={native.SharedM3Updates-startM3} action_magnitude_variants={actionMagnitudes.Count} bias={native.BiasStrength:R} m2_loss={native.LastM2Loss:R} m3_loss={native.LastM3Loss:R} control_failures={native.ControlFailures}");
    }

    [UnityTest]
    [Explicit("Manual 20,000-tick v18 comparison harness; excluded from routine validation")]
    [Timeout(420000)]
    public IEnumerator V18ManualComparisonRunsTwentyThousandTicksWithInteractiveFeaturesOff()
    {
        SceneManager.LoadScene("SampleScene", LoadSceneMode.Single);
        yield return null;
        NativeEcosystemController native = null;
        float setupDeadline = Time.realtimeSinceStartup + 15f;
        while (Time.realtimeSinceStartup < setupDeadline) {
            native = Object.FindAnyObjectByType<NativeEcosystemController>();
            if (native != null && native.Population == native.populationTarget) break;
            yield return null;
        }
        Assert.NotNull(native);
        Assert.IsFalse(native.M1Enabled);
        Assert.IsFalse(native.PopulationDynamicsEnabled);
        typeof(NativeEcosystemController).GetField("nextCheckpoint", BindingFlags.Instance | BindingFlags.NonPublic)?.SetValue(native, float.PositiveInfinity);
        native.checkpointIntervalSeconds = float.PositiveInfinity;

        int startTick = native.Tick;
        int startM3 = native.SharedM3Updates;
        long startMemory = System.GC.GetTotalMemory(false);
        var plannerSamples = new List<float>();
        var m3Samples = new List<float>();
        var timer = Stopwatch.StartNew();
        int nextCurveTick = 2000;
        Time.timeScale = 100f;
        float deadline = Time.realtimeSinceStartup + 390f;
        while (native != null && native.Tick - startTick < 20000 && Time.realtimeSinceStartup < deadline) {
            plannerSamples.Add(native.PlannerMilliseconds);
            m3Samples.Add(native.LastM3Loss);
            if (native.Tick - startTick >= nextCurveTick) {
                Debug.Log($"V18_COMPARISON_CURVE tick={native.Tick-startTick} m2={native.LastM2Loss:R} m3={native.LastM3Loss:R} m3_updates={native.SharedM3Updates-startM3} fps={native.RenderRateHz:R} planner_ms={native.PlannerMilliseconds:R} root_speed={native.MeanRootSpeed:R}");
                nextCurveTick += 2000;
            }
            yield return null;
        }
        timer.Stop();

        Assert.NotNull(native);
        Assert.GreaterOrEqual(native.Tick - startTick, 20000, "v18 comparison run did not complete before the real-time deadline");
        Assert.IsFalse(native.M1Enabled);
        Assert.IsFalse(native.PopulationDynamicsEnabled);
        Assert.AreEqual(0, native.ImmediateReplacementCount);
        Assert.AreEqual(0, native.ControlFailures, native.LastControlError);
        plannerSamples.Sort();
        float p50 = plannerSamples.Count == 0 ? 0f : plannerSamples[plannerSamples.Count / 2];
        float p95 = plannerSamples.Count == 0 ? 0f : plannerSamples[Mathf.Min(plannerSamples.Count - 1, Mathf.FloorToInt(plannerSamples.Count * .95f))];
        long endMemory = System.GC.GetTotalMemory(false);
        Debug.Log($"V18_COMPARISON_FINAL tick_delta={native.Tick-startTick} seconds={timer.Elapsed.TotalSeconds:R} ticks_per_second={(native.Tick-startTick)/timer.Elapsed.TotalSeconds:R} render_fps={native.RenderRateHz:R} planner_p50_ms={p50:R} planner_p95_ms={p95:R} m3_updates={native.SharedM3Updates-startM3} m2_loss={native.LastM2Loss:R} m3_loss={native.LastM3Loss:R} root_speed={native.MeanRootSpeed:R} managed_memory_delta={endMemory-startMemory} checkpoint_status={native.CheckpointStatus}");
    }
}
