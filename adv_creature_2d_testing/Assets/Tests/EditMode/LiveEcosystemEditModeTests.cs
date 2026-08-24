using System;
using System.Collections.Generic;
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
        float[] eight = NativeCreatureModel.PeriodicPhase(8);
        float[] sixtyFour = NativeCreatureModel.PeriodicPhase(64);
        Assert.AreEqual(8, zero.Length);
        Assert.AreEqual(0f, zero[0], .0001f);
        Assert.AreEqual(1f, zero[1], .0001f);
        Assert.AreEqual(zero[0], eight[0], .0001f);
        Assert.AreEqual(zero[1], eight[1], .0001f);
        for (int i = 0; i < zero.Length; i++) Assert.AreEqual(zero[i], sixtyFour[i], .0001f);
    }

    [Test]
    public void M2WeightSchemaIncludesPhaseInputAndFourthDenseLayer()
    {
        NativeBrainWeights weights = NativeBrainWeights.Create(1);
        Assert.AreEqual(21 * 16, weights.m2Input.Length);
        Assert.AreEqual(32 * 16, weights.m2Middle.Length);
        Assert.AreEqual(16 * 16, weights.m2Deep.Length);
        Assert.AreEqual(16 * 16, weights.m2Extra.Length);
        Assert.AreEqual(16 * 16 * 3, weights.m2History.Length);
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
