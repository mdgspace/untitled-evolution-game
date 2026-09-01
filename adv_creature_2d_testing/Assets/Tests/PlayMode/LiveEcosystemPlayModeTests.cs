using System.Collections;
using System.Linq;
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
            if (native != null && native.Population == 6) break;
            yield return null;
        }
        Assert.NotNull(native);
        Assert.AreEqual(0, Object.FindObjectsByType<PythonBridge>(FindObjectsInactive.Exclude).Count(component => component.isActiveAndEnabled));
        Assert.AreEqual(75f,native.evolutionIntervalSeconds);
        Assert.AreEqual(12,native.bodyMutationCooldownGenerations);
        EnvironmentSpawner environment = Object.FindAnyObjectByType<EnvironmentSpawner>();
        Assert.NotNull(environment);
        Assert.AreEqual(1, Object.FindObjectsByType<EnvironmentSpawner>(FindObjectsInactive.Exclude).Length);
        Assert.AreEqual(6, native.Population, native.LastCheckpointError);
        Assert.NotNull(Object.FindAnyObjectByType<ObserverCamera>());
        Assert.NotNull(GameObject.Find("WorldWallLeft"));
        Assert.NotNull(GameObject.Find("WorldWallRight"));
        Food[] food = Object.FindObjectsByType<Food>(FindObjectsInactive.Include);
        Predator[] predators = Object.FindObjectsByType<Predator>(FindObjectsInactive.Include);
        Assert.IsFalse(native.M1Enabled);Assert.AreEqual("locked at startup",native.M1ActivationSource);Assert.IsFalse(environment.FeaturesEnabled);Assert.AreEqual(0,food.Length);Assert.AreEqual(0,predators.Length);
        yield return new WaitForFixedUpdate();Assert.IsFalse(native.M1Enabled);Assert.AreEqual("locked at startup",native.M1ActivationSource);
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
        Assert.Greater(actionMagnitudes.Count, 2, "Executed action magnitudes lacked diversity");
        Assert.AreEqual(0f, native.BiasStrength, 0f, "Bias must be exactly zero at/after the scaled cutoff");
        Assert.AreEqual(0, native.ControlFailures, native.LastControlError);
        Assert.IsFalse(float.IsNaN(native.LastM2Loss) || float.IsInfinity(native.LastM2Loss));
        Assert.IsFalse(float.IsNaN(native.LastM3Loss) || float.IsInfinity(native.LastM3Loss));
        Assert.That(native.SequenceStep, Is.InRange(0, NativeCreatureModel.PlanningHorizon));
        Debug.Log($"OPEN_LOOP_SCALED tick_delta={native.Tick-startTick} population={native.Population} replacements={native.ImmediateReplacementCount-startReplacements} m3_updates={native.SharedM3Updates-startM3} action_magnitude_variants={actionMagnitudes.Count} bias={native.BiasStrength:R} m2_loss={native.LastM2Loss:R} m3_loss={native.LastM3Loss:R} control_failures={native.ControlFailures}");
    }
}
