using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using NUnit.Framework;
using Newtonsoft.Json.Linq;
using UnityEngine;
using Object = UnityEngine.Object;

public class LiveEcosystemEditModeTests
{
    [Test]
    public void BodyGenomeValidationRejectsDuplicateInnovations()
    {
        JObject raw = JObject.Parse(@"{
            'genome_id':'fixture', 'torso_width':1, 'torso_height':1,
            'limbs':[
              {'innovation_id':1,'parent_innovation_id':0,'attachment_slot':0,'joint_type':'hinge'},
              {'innovation_id':1,'parent_innovation_id':0,'attachment_slot':1,'joint_type':'hinge'}
            ]}");
        BodyGenomeDto genome = BodyGenomeDto.FromJson(raw);
        Assert.Throws<ArgumentException>(() => genome.Validate());
    }

    [Test]
    public void FourSpawnSlotsAreUniqueAndInsideWorld()
    {
        HashSet<Vector2> positions = new HashSet<Vector2>();
        for (int slot = 0; slot < 4; slot++) {
            Vector2 position = WorldLayout.CreatureSpawnPosition(slot);
            Assert.That(position.x, Is.InRange(-WorldLayout.WorldHalfWidth, WorldLayout.WorldHalfWidth));
            Assert.IsTrue(positions.Add(position));
        }
        Assert.AreEqual(4, positions.Count);
    }

    [Test]
    public void PeriodicPhaseHasExpectedDimensionsAndPeriods()
    {
        float[] zero = NativeCreatureModel.PeriodicPhase(0);
        float[] four = NativeCreatureModel.PeriodicPhase(4);
        float[] thirtyTwo = NativeCreatureModel.PeriodicPhase(32);
        Assert.AreEqual(8, zero.Length);
        Assert.AreEqual(0f, zero[0], .0001f);
        Assert.AreEqual(1f, zero[1], .0001f);
        Assert.AreEqual(zero[0], four[0], .0001f);
        Assert.AreEqual(zero[1], four[1], .0001f);
        for (int i = 0; i < zero.Length; i++) Assert.AreEqual(zero[i], thirtyTwo[i], .0001f);
    }

    [Test]
    public void M2AndM3WeightSchemaIncludesTheDeepenedNativeLayers()
    {
        NativeBrainWeights weights = NativeBrainWeights.Create(1);
        Assert.AreEqual(21 * 16, weights.m2Input.Length);
        Assert.AreEqual(32 * 16, weights.m2Middle.Length);
        Assert.AreEqual(16 * 16, weights.m2Deep.Length);
        Assert.AreEqual(16 * 16, weights.m2Extra.Length);
        Assert.AreEqual(16 * 16, weights.m2Ultra.Length);
        Assert.AreEqual(16 * 16, weights.m2Final.Length);
        Assert.AreEqual(16 * 16 * 3, weights.m2History.Length);
        Assert.AreEqual(16 * 16, weights.m3Deep.Length);
        Assert.AreEqual(16 * 16, weights.m3Extra.Length);
        Assert.AreEqual(16 * NativeCreatureModel.PlanningHorizon, weights.m2Output.Length);
        Assert.AreEqual((16 + NativeCreatureModel.ConsequenceSize) * NativeCreatureModel.PlanningHorizon, weights.m2Refine.Length);
        Assert.AreEqual(16 * NativeCreatureModel.M3OutputSize, weights.m3Output.Length);
        Assert.AreEqual(NativeCreatureModel.M3OutputSize, weights.m3Action.Length);
    }

    [Test]
    public void RecedingHorizonPlanHasFiveStepsAndOnlyValidLimbsReceiveActions()
    {
        CreateFixtureCreature(out GameObject root, out _, out CreatureBrain brain);
        try
        {
            NativeCreatureModel model = new NativeCreatureModel(NativeBrainWeights.Create(3), 4);
            NativeActionPlan plan = model.PlanActions(new float[28], brain.allLimbs.ToArray(), 1, false, .5f, false);
            Assert.AreEqual(NativeCreatureModel.PlanningHorizon, plan.actions.GetLength(0));
            Assert.AreEqual(NativeCreatureModel.MaxLimbs, plan.actions.GetLength(1));
            Assert.That(Mathf.Abs(plan.actions[0, 0]), Is.InRange(NativeCreatureModel.MinimumActionDegrees, NativeCreatureModel.MaximumActionDegrees));
            Assert.AreEqual(plan.actions[0, 0], plan.FirstAction()[0]);
            for (int slot = 1; slot < NativeCreatureModel.MaxLimbs; slot++) Assert.AreEqual(0f, plan.actions[0, slot]);
            Assert.AreEqual(NativeCreatureModel.GoalWindowTicks - NativeCreatureModel.ActionTicks, model.GoalTicksRemaining);
            for (int action = 0; action < 4; action++) model.PlanActions(new float[28], brain.allLimbs.ToArray(), action + 2, false, .5f, false);
            Assert.AreEqual(0, model.GoalTicksRemaining);
            model.PlanActions(new float[28], brain.allLimbs.ToArray(), 7, false, .5f, false);
            Assert.AreEqual(NativeCreatureModel.GoalWindowTicks - NativeCreatureModel.ActionTicks, model.GoalTicksRemaining);
        }
        finally { UnityEngine.Object.DestroyImmediate(root); }
    }

    [Test]
    public void LatePlanWeightsFavorTheFifthPredictedAction()
    {
        float[] weights = NativeCreatureModel.LatePlanWeights();
        Assert.AreEqual(NativeCreatureModel.PlanningHorizon, weights.Length);
        Assert.AreEqual(1f, weights.Sum(), .0001f);
        Assert.Less(weights[0], weights[4]);
        Assert.AreEqual(25f / 55f, weights[4], .0001f);
    }

    [Test]
    public void RealTwoTickTransitionTrainsM3AndRestoresPlannerState()
    {
        CreateFixtureCreature(out GameObject root, out _, out CreatureBrain brain);
        try
        {
            NativeCreatureModel model = new NativeCreatureModel(NativeBrainWeights.Create(5), 6);
            model.PlanActions(new float[28], brain.allLimbs.ToArray(), 1, false, .5f, false);
            Assert.IsTrue(model.TrainDynamics(new NativeTransitionTarget { displacement = new Vector2(.02f, 0f), energyDelta = -.01f, torsoGrounded = false, predatorContact = false }));
            Assert.AreEqual(1, model.DynamicsReplayCount);
            Assert.IsFalse(float.IsNaN(model.LastM3Loss));

            NativeCreatureSave saved = model.Capture();
            NativeCreatureModel restored = new NativeCreatureModel(NativeBrainWeights.Create(7), 8);
            restored.Restore(saved);
            Assert.AreEqual(model.GoalTicksRemaining, restored.GoalTicksRemaining);
            Assert.AreEqual(1, restored.DynamicsReplayCount);
        }
        finally { UnityEngine.Object.DestroyImmediate(root); }
    }

    [Test]
    public void RecedingHorizonCheckpointFingerprintRejectsThePriorOneStepSchema()
    {
        const string previousFingerprint = "m1gru16-front-m2gru16-phase8-16-32-64-d4-m3gru16-d2-replayfloats-h4-l8-p4-a8-lossnorm";
        Assert.AreNotEqual(previousFingerprint, NativeEcosystemCheckpoint.ConfigurationFingerprint);
        StringAssert.Contains("m2h5-k3", NativeEcosystemCheckpoint.ConfigurationFingerprint);
    }

    [Test]
    public void NewbornStationaryCreatureHasAnExplorationWindowBeforeIdleStarvation()
    {
        GameObject torsoObject = new GameObject("torso-idle-budget");
        try
        {
            Torso torso = torsoObject.AddComponent<Torso>();
            Assert.AreEqual(2.5f, torso.idleEnergyPerSecond, .0001f);
            Assert.AreEqual(4f, torso.idleAfterSeconds, .0001f);
        }
        finally { UnityEngine.Object.DestroyImmediate(torsoObject); }
    }

    [Test]
    public void ReplaySerializationUsesPrimitiveCoordinates()
    {
        string json = Newtonsoft.Json.JsonConvert.SerializeObject(new NativeDynamicsReplaySample { combined = new float[16], predictionX = 1f, predictionY = 2f, measuredX = 3f, measuredY = 4f });
        StringAssert.DoesNotContain("normalized", json);
        StringAssert.Contains("predictionX", json);
    }

    [Test]
    public void ActionLeaseIsBoundedAndDoesNotSilentlyExpireToZero()
    {
        GameObject parentObject = new GameObject("parent");
        GameObject limbObject = new GameObject("limb");
        try {
            Rigidbody2D parent = parentObject.AddComponent<Rigidbody2D>();
            CreatureIdentity identity = parentObject.AddComponent<CreatureIdentity>();
            Limb limb = limbObject.AddComponent<Limb>();
            LimbGeneDto gene = new LimbGeneDto { innovation_id = 1, parent_innovation_id = 0,
                                                 attachment_slot = 3, max_torque = 50 };
            limb.InitFromGene(parent, Vector2.zero, Vector2.right, gene, new[] { 3 }, identity, .45f);
            limb.ApplyIntervalDelta(100f, 3, .02f);
            Assert.AreEqual(NativeCreatureModel.MaximumActionDegrees, limb.ExecutedIntervalDelta, .0001f);
            Assert.AreEqual(NativeCreatureModel.MaximumActionDegrees / 3f, limb.lastAppliedDelta, .0001f);
            limb.AdvanceMotorTick(); limb.AdvanceMotorTick(); limb.AdvanceMotorTick(); limb.AdvanceMotorTick();
            Assert.AreEqual(NativeCreatureModel.MaximumActionDegrees / 3f, limb.lastAppliedDelta, .0001f,
                "The last learned command must remain active while an asynchronous response is late.");
        } finally {
            UnityEngine.Object.DestroyImmediate(limbObject);
            UnityEngine.Object.DestroyImmediate(parentObject);
        }
    }

    [Test]
    public void KillImmediatelyDisablesTheWholePhenotype()
    {
        CreateFixtureCreature(out GameObject root, out Torso torso, out CreatureBrain brain);
        try {
            brain.Kill("test");
            Assert.IsFalse(torso.Rigidbody.simulated);
            Assert.IsFalse(torso.GetComponent<Collider2D>().enabled);
            foreach (Limb limb in brain.allLimbs) {
                Assert.IsFalse(limb.Rigidbody.simulated);
                Assert.IsFalse(limb.GetComponent<Collider2D>().enabled);
            }
        } finally { UnityEngine.Object.DestroyImmediate(root); }
    }

    [Test]
    public void ControllerFaultPausesOnlyAfterTheOneSecondHeldActionLimit()
    {
        GameObject bridgeObject = new GameObject("bridge");
        try {
            PythonBridge bridge = bridgeObject.AddComponent<PythonBridge>();
            SetPrivate(bridge, "receivedInitialWorld", true);
            SetPrivate(bridge, "heldActionStartedTick", 0);
            SetPrivate(bridge, "tick", 50);
            typeof(PythonBridge).GetMethod("RunFixedSimulationStep", BindingFlags.Instance | BindingFlags.NonPublic)
                .Invoke(bridge, null);
            Assert.IsTrue(bridge.ControllerFaultPaused);
            StringAssert.Contains("held action", bridge.PipelineStatus);
        } finally { UnityEngine.Object.DestroyImmediate(bridgeObject); }
    }

    [Test]
    public void PredatorRequiresThreeConsecutiveContactTicks()
    {
        CreateFixtureCreature(out GameObject root, out Torso torso, out CreatureBrain brain);
        GameObject predatorObject = new GameObject("predator");
        try {
            Predator predator = predatorObject.AddComponent<Predator>();
            MethodInfo enter = typeof(Predator).GetMethod("OnTriggerEnter2D", BindingFlags.Instance | BindingFlags.NonPublic);
            MethodInfo fixedUpdate = typeof(Predator).GetMethod("FixedUpdate", BindingFlags.Instance | BindingFlags.NonPublic);
            enter.Invoke(predator, new object[] { torso.GetComponent<Collider2D>() });
            fixedUpdate.Invoke(predator, null); fixedUpdate.Invoke(predator, null);
            Assert.IsFalse(brain.IsDead);
            fixedUpdate.Invoke(predator, null);
            Assert.IsTrue(brain.IsDead);
            Assert.AreEqual("predator", brain.DeathReason);
        } finally {
            UnityEngine.Object.DestroyImmediate(predatorObject);
            UnityEngine.Object.DestroyImmediate(root);
        }
    }

    private static void CreateFixtureCreature(out GameObject root, out Torso torso, out CreatureBrain brain)
    {
        root = new GameObject("fixture-root");
        CreatureIdentity identity = root.AddComponent<CreatureIdentity>();
        identity.creatureId = "fixture"; identity.speciesId = "fixture-species";
        GameObject torsoObject = new GameObject("torso"); torsoObject.transform.SetParent(root.transform);
        torso = torsoObject.AddComponent<Torso>();
        BodyGenomeDto genome = new BodyGenomeDto { genome_id = "fixture", torso_width = 1, torso_height = 1,
            limbs = new List<LimbGeneDto> { new LimbGeneDto { innovation_id = 1, parent_innovation_id = 0,
                                                               attachment_slot = 3, joint_type = "hinge" } } };
        torso.InitFromGenome(identity, genome); identity.torso = torso;
        brain = torsoObject.AddComponent<CreatureBrain>(); brain.Init(torso, torso.GetAllLimbs());
    }

    private static void SetPrivate(object target, string field, object value)
    {
        typeof(PythonBridge).GetField(field, BindingFlags.Instance | BindingFlags.NonPublic).SetValue(target, value);
    }
}
