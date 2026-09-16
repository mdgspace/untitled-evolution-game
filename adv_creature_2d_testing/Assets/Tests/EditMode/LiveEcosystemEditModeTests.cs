using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using Newtonsoft.Json;
using NUnit.Framework;
using UnityEngine;

public class LiveEcosystemEditModeTests
{
    [Test]
    public void V24WidthsDepthsAndHeadPartitionsAreExact()
    {
        Assert.AreEqual(32,NativeCreatureModel.M1Width);Assert.AreEqual(36,NativeCreatureModel.M2Width);Assert.AreEqual(36,NativeCreatureModel.M3Width);
        Assert.AreEqual(2,NativeCreatureModel.M2TransformerBlocks);Assert.AreEqual(2,NativeCreatureModel.M3TransformerBlocks);Assert.AreEqual(4,NativeCreatureModel.M2AttentionHeads);Assert.AreEqual(4,NativeCreatureModel.M3AttentionHeads);Assert.AreEqual(9,NativeCreatureModel.M2Width/NativeCreatureModel.M2AttentionHeads);
        Assert.AreEqual(8,NativeCreatureModel.M2MlpHiddenLayers);Assert.AreEqual(6,NativeCreatureModel.M3MlpHiddenLayers);Assert.AreEqual(22,NativeCreatureModel.M2ContextSize);Assert.AreEqual(18,NativeCreatureModel.M3ContextSize);
        NativeBrainWeights policy=NativeBrainWeights.Create(1);NativeDynamicsWeights dynamics=NativeDynamicsWeights.Create(2);
        Assert.AreEqual(28*32,policy.m1Input.Length);Assert.AreEqual(22*36,policy.m2Context.Length);Assert.AreEqual(NativeCreatureModel.JointFeatureSize*36,policy.m2Joint.Length);Assert.AreEqual(24*36,policy.m2Position.Length);Assert.AreEqual(36*36,policy.m2Deep4.Length);Assert.AreEqual(NativeCreatureModel.PlanningHorizon*36,policy.m2StepEmbedding.Length);Assert.AreEqual(36,policy.m2Output.Length);
        Assert.AreEqual(NativeCreatureModel.M3JointInputSize*36,dynamics.joint.Length);Assert.AreEqual(NativeCreatureModel.PlanningHorizon*36,dynamics.horizonQueries.Length);Assert.AreEqual(36*36,dynamics.deep3.Length);Assert.AreEqual(36*NativeCreatureModel.M3ValuesPerHorizon,dynamics.output.Length);
    }

    [Test]
    public void PolicyOwnsOnlyM1M2AndDynamicsIsSingleSeparateType()
    {
        string policyFields=string.Join(" ",typeof(NativeBrainWeights).GetFields().Select(f=>f.Name)).ToLowerInvariant();StringAssert.DoesNotContain("m3",policyFields);StringAssert.DoesNotContain("recurrent",policyFields);StringAssert.DoesNotContain("gru",policyFields);
        Assert.IsTrue(typeof(NativeCreatureCheckpoint).GetField("m3").FieldType==typeof(NativeM3LearningState));
    }

    [Test]
    public void ContactAndOrientationJointFeaturesAreExact()
    {
        float[] f=NativeCreatureModel.JointFeatures(90f,360f,true,false,true);Assert.AreEqual(NativeCreatureModel.JointFeatureSize,f.Length);Assert.AreEqual(1f,f[0],1e-5f);Assert.AreEqual(0f,f[1],1e-5f);Assert.AreEqual(0f,f[2],1e-5f);Assert.AreEqual(-.5f,f[3],1e-5f);CollectionAssert.AreEqual(new[]{1f,0f,1f},f.Skip(4).Take(3).ToArray());
    }

    [Test]
    public void TopologyEncodingIsDeterministicUniformAndIdentityFree()
    {
        float[] a=NativeCreatureModel.JointPositionEncoding(new[]{5,3},99),b=NativeCreatureModel.JointPositionEncoding(new[]{5,3},-7),c=NativeCreatureModel.JointPositionEncoding(new[]{5,2},99);Assert.AreEqual(24,a.Length);CollectionAssert.AreEqual(a,b);Assert.Greater(Difference(a,c),1e-4f);Assert.AreEqual(1f,a[0]);Assert.AreEqual(1f,a[5]);Assert.AreEqual(0f,a[10]);Assert.AreEqual(2f/3f,a[15],1e-6f);
    }

    [Test]
    public void PaddedLimbFeaturesAndActionsAreMasked()
    {
        NativeObservation clean=ObservationFixture(10000),padded=JsonConvert.DeserializeObject<NativeObservation>(JsonConvert.SerializeObject(clean));int limb=NativeCreatureModel.MaxLimbs-1;for(int i=0;i<NativeCreatureModel.JointFeatureSize;i++)padded.jointFeatures[limb*NativeCreatureModel.JointFeatureSize+i]=1000+i;for(int i=0;i<NativeCreatureModel.PositionEncodingSize;i++)padded.positionEncodings[limb*NativeCreatureModel.PositionEncodingSize+i]=-1000-i;NativeBrainWeights policy=NativeBrainWeights.Create(3);NativeDynamicsWeights dynamics=NativeDynamicsWeights.Create(4);Assert.AreEqual(NativeCreatureModel.M2LossForTesting(clean,policy,dynamics),NativeCreatureModel.M2LossForTesting(padded,policy,dynamics),1e-6f);NativeCompletedSequence first=SequenceFixture("a"),second=CloneSequence(first);for(int s=0;s<NativeCreatureModel.PlanningHorizon;s++)second.actions[s*NativeCreatureModel.MaxLimbs+limb]=999f;Assert.AreEqual(NativeCreatureModel.M3LossForTesting(first,dynamics),NativeCreatureModel.M3LossForTesting(second,dynamics),1e-6f);
    }

    [Test]
    public void SiLUAndRmsNormGradientsMatchCentralDifference()
    {
        float x=.37f,epsilon=1e-3f;float numerical=(NativeCreatureModel.SiLU(x+epsilon)-NativeCreatureModel.SiLU(x-epsilon))/(2f*epsilon);Assert.AreEqual(numerical,NativeCreatureModel.SiLUDerivative(x),1e-4f);
        int width=NativeCreatureModel.M2Width;float[] input=Enumerable.Range(0,width).Select(i=>(i-(width/2-1f))/80f).ToArray(),gamma=Enumerable.Repeat(1f,width).ToArray(),upstream=Enumerable.Range(0,width).Select(i=>Mathf.Sin(i*.13f)).ToArray();float[] analytic=NativeCreatureModel.RmsNormInputGradientForTesting(input,gamma,upstream);int index=width/2;float original=input[index];input[index]=original+epsilon;float plus=Dot(NativeCreatureModel.RmsNormForTesting(input,gamma),upstream);input[index]=original-epsilon;float minus=Dot(NativeCreatureModel.RmsNormForTesting(input,gamma),upstream);AssertRelative(analytic[index],(plus-minus)/(2f*epsilon),.002f);
    }

    [Test]
    public void M3PoseAndAuxiliaryGradientsReachSharedOutputAndQueries()
    {
        NativeDynamicsWeights dynamics=NativeDynamicsWeights.Create(11);NativeCompletedSequence sample=SequenceFixture("gradient");float[] output=NativeCreatureModel.M3GradientForTesting(sample,dynamics,"output"),queries=NativeCreatureModel.M3GradientForTesting(sample,dynamics,"horizonQueries");Assert.Greater(output.Sum(Mathf.Abs),1e-6f);Assert.Greater(queries.Sum(Mathf.Abs),1e-6f);
        int index=Largest(output);float original=dynamics.output[index],epsilon=5e-4f;dynamics.output[index]=original+epsilon;float plus=NativeCreatureModel.M3LossForTesting(sample,dynamics);dynamics.output[index]=original-epsilon;float minus=NativeCreatureModel.M3LossForTesting(sample,dynamics);AssertRelative(output[index],(plus-minus)/(2f*epsilon),.05f);
    }

    [Test]
    public void M3AuxiliaryLossUsesEndpointMseAndPerLimbContactBce()
    {
        float[] predicted=new float[NativeCreatureModel.M3AuxiliaryOutputSize],actual=new float[NativeCreatureModel.M3AuxiliaryOutputSize];for(int s=0;s<NativeCreatureModel.PlanningHorizon;s++){int i=s*NativeCreatureModel.AuxiliaryValueCount;predicted[i]=.5f;actual[i]=.1f;int contactOffset=i+NativeCreatureModel.MaxLimbs*NativeCreatureModel.EndpointValueCount;predicted[contactOffset]=.5f;actual[contactOffset]=1f;}float total=NativeCreatureModel.M3AuxiliaryLoss(predicted,actual,null,1,out float endpoint,out float contact);Assert.Greater(endpoint,0f);Assert.Greater(contact,0f);Assert.AreEqual(endpoint+contact,total,1e-7f);Assert.IsTrue(NativeCreatureModel.IsFinite(total));
    }

    [Test]
    public void BoundaryBatchIsOneOrderIndependentAdamStep()
    {
        var a=SequenceFixture("a");var b=SequenceFixture("b");b.actualPoses[0]+=.3f;NativeDynamicsWeights left=NativeDynamicsWeights.Create(20),right=left.Clone();var lo=new NativeOptimizerState();var ro=new NativeOptimizerState();NativeM3BatchMetrics lm=NativeCreatureModel.TrainSharedM3Batch(new[]{a,b},left,lo),rm=NativeCreatureModel.TrainSharedM3Batch(new[]{b,a},right,ro);Assert.AreEqual(2,lm.sampleCount);Assert.AreEqual(1,lm.optimizerSteps);Assert.AreEqual(1,lo.step);Assert.AreEqual(0f,Difference(left.output,right.output),1e-7f);Assert.AreEqual(lm.loss,rm.loss,1e-7f);
    }

    [Test]
    public void PerCreatureM3StateCloneIsIndependentAndUsesLocalReplayBudget()
    {
        var source=new NativeM3LearningState{dynamics=NativeDynamicsWeights.Create(90)};source.replay.Offer(SequenceFixture("source"));var child=source.Clone();child.dynamics.output[0]+=.5f;child.replay.Offer(SequenceFixture("child"));Assert.AreNotEqual(source.dynamics.output[0],child.dynamics.output[0]);Assert.AreEqual(1,source.replay.Count);Assert.AreEqual(2,child.replay.Count);Assert.AreEqual(12288,NativeReplayBuffer.Capacity);
    }

    [Test]
    public void M2GradientFlowsThroughFrozenM3AndSharedStepDecoder()
    {
        NativeBrainWeights policy=NativeBrainWeights.Create(30);NativeDynamicsWeights dynamics=NativeDynamicsWeights.Create(31),before=dynamics.Clone();NativeObservation observation=ObservationFixture(10000);float[] gradient=NativeCreatureModel.M2GradientForTesting(observation,policy,dynamics,"stepEmbedding");Assert.Greater(gradient.Sum(Mathf.Abs),1e-6f);var model=new NativeCreatureModel(policy,32);NativeActionSequence sequence=model.OptimizeAndPlan(observation,dynamics,100);Assert.AreEqual(1,model.M2Optimizer.step);Assert.AreEqual(0f,Difference(before.output,dynamics.output));for(int s=0;s<NativeCreatureModel.PlanningHorizon;s++)for(int l=0;l<sequence.limbCount;l++)Assert.That(sequence.actions[s,l],Is.InRange(-8f,8f));
    }

    [Test]
    public void M2EmitsSevenActionsAndM3ConsumesTheEntireActionHorizon()
    {
        NativeObservation observation=ObservationFixture(10000);NativeDynamicsWeights dynamics=NativeDynamicsWeights.Create(33);var model=new NativeCreatureModel(NativeBrainWeights.Create(34),35);NativeActionSequence sequence=model.Plan(observation,dynamics,100,NativeCreatureModel.AssistedBiasStrength,0f);Assert.AreEqual(NativeCreatureModel.PlanningHorizon,sequence.actions.GetLength(0));Assert.AreEqual(NativeCreatureModel.MaxLimbs,sequence.actions.GetLength(1));float[,] changed=(float[,])sequence.actions.Clone();changed[NativeCreatureModel.PlanningHorizon-1,0]=sequence.actions[NativeCreatureModel.PlanningHorizon-1,0]>=0f?-8f:8f;Assert.Greater(Difference(NativeCreatureModel.PredictM3ForTesting(observation,sequence.actions,dynamics),NativeCreatureModel.PredictM3ForTesting(observation,changed,dynamics)),1e-7f);
    }

    [Test]
    public void CurriculumSoftBiasIsAppliedBeforeLocomotionIsVerified()
    {
        NativeObservation observation=ObservationFixture(10000);NativeBrainWeights policy=NativeBrainWeights.Create(36);foreach(KeyValuePair<string,float[]> parameter in policy.M2Parameters())Array.Clear(parameter.Value,0,parameter.Value.Length);var model=new NativeCreatureModel(policy,37);NativeActionSequence sequence=model.Plan(observation,NativeDynamicsWeights.Create(38),0,NativeCreatureModel.AssistedBiasStrength,0f);Assert.AreEqual(NativeCreatureModel.AssistedBiasStrength,sequence.biasStrength,1e-7f);Assert.Greater(sequence.actions.Cast<float>().Sum(Mathf.Abs),1e-3f);
    }

    [Test]
    public void MutationScalingMatchesFallbackWidth()
    {
        Assert.AreEqual(1f,NativeCreatureModel.M2MutationScale);Assert.Greater(NativeBrainWeights.Create(40).M2Parameters().Sum(p=>p.Value.Length),0);
    }

    [Test]
    public void PoseLossWeightsAndCadenceRemainExact()
    {
        Assert.AreEqual(5,NativeCreatureModel.ActionTicks);Assert.AreEqual(35,NativeCreatureModel.SequenceTicks);Assert.AreEqual(35,NativeCreatureModel.M1GoalWindowTicks);Assert.AreEqual(1000,NativeCreatureModel.GoalWindowTicks);CollectionAssert.AreEqual(new[]{1f,1f,2f,4f,8f,12f,16f},NativeCreatureModel.GoalStepWeights);Assert.Less(NativeCreatureModel.M2HeadingLossWeight,1f);float[] p=new float[NativeCreatureModel.M3OutputSize];for(int s=0;s<NativeCreatureModel.PlanningHorizon;s++)p[s*NativeCreatureModel.PoseValueCount+3]=1f;p[0]=1f;float early=NativeCreatureModel.M2GoalLoss(p,Vector2.zero,Vector2.up);p=new float[NativeCreatureModel.M3OutputSize];for(int s=0;s<NativeCreatureModel.PlanningHorizon;s++)p[s*NativeCreatureModel.PoseValueCount+3]=1f;p[NativeCreatureModel.M3OutputSize-NativeCreatureModel.PoseValueCount]=1f;Assert.AreEqual(16f*early,NativeCreatureModel.M2GoalLoss(p,Vector2.zero,Vector2.up),1e-6f);
    }

    [Test]
    public void M2UsesAReachableGoalDirectedHorizonInsteadOfTheFullArenaDistance()
    {
        Vector2 right=NativeCreatureModel.PlanningGoal(new Vector2(40f,0f)),left=NativeCreatureModel.PlanningGoal(new Vector2(-60f,0f));Assert.AreEqual(NativeCreatureModel.M2PlanningGoalDisplacement,right.magnitude,1e-6f);Assert.AreEqual(-NativeCreatureModel.M2PlanningGoalDisplacement,left.x,1e-6f);Assert.AreEqual(NativeCreatureModel.M2GoalLoss(new float[NativeCreatureModel.M3OutputSize],new Vector2(40f,0f),Vector2.up),NativeCreatureModel.M2GoalLoss(new float[NativeCreatureModel.M3OutputSize],new Vector2(60f,0f),Vector2.up),1e-6f);
    }

    [Test]
    public void PositionTelemetryRetainsRelativeErrorWithoutOptimizingIt()
    {
        float[] actual=new float[NativeCreatureModel.M3OutputSize];for(int s=0;s<NativeCreatureModel.PlanningHorizon;s++)actual[s*NativeCreatureModel.PoseValueCount+3]=1f;actual[NativeCreatureModel.M3OutputSize-NativeCreatureModel.PoseValueCount]=.1f;float zero=NativeCreatureModel.M3Loss(new float[NativeCreatureModel.M3OutputSize],actual,null,out float mse,out float relative),correct=NativeCreatureModel.M3Loss((float[])actual.Clone(),actual,null,out _,out _);Assert.Greater(relative,0f);Assert.AreEqual(0f,correct,1e-7f);Assert.IsTrue(NativeCreatureModel.IsFinite(zero));
    }

    [Test]
    public void BinaryTensorStoreRoundTripsAndDetectsCorruption()
    {
        string directory=Path.Combine(Path.GetTempPath(),"native-v26-test-"+Guid.NewGuid().ToString("N"));Directory.CreateDirectory(directory);string file=Path.Combine(directory,"native_ecosystem_v26.test.bin");try{var checkpoint=CheckpointFixture();checkpoint.sharedLearning=new NativeSharedLearningState{m2=NativeBrainWeights.Create(81),m3=new NativeM3LearningState{dynamics=NativeDynamicsWeights.Create(82)}};checkpoint.sharedLearning.m3.usefulReplay.Offer(SequenceFixture("useful"));checkpoint.movementLearning=new NativeMovementLearningState{weights=NativeMovementWeights.Create(83),qualifiedWeights=NativeMovementWeights.Create(84),qualifiedOptimizer=new NativeOptimizerState(),simpleBodyQualified=true};NativeTensorCheckpointStore.Write(file,checkpoint);string hash=NativeTensorCheckpointStore.Sha256(file);string json=JsonConvert.SerializeObject(checkpoint);StringAssert.DoesNotContain("m2Context",json);StringAssert.DoesNotContain("nodeInput",json);StringAssert.DoesNotContain("\"slots\"",json);var metadata=JsonConvert.DeserializeObject<NativeEcosystemCheckpoint>(json);NativeTensorCheckpointStore.Read(file,metadata);Assert.AreEqual(checkpoint.sharedLearning.m3.dynamics.output[10],metadata.sharedLearning.m3.dynamics.output[10]);Assert.AreEqual(checkpoint.sharedLearning.m2.m2Context[10],metadata.sharedLearning.m2.m2Context[10]);Assert.AreEqual(checkpoint.movementLearning.weights.nodeInput[10],metadata.movementLearning.weights.nodeInput[10]);Assert.AreEqual(checkpoint.movementLearning.qualifiedWeights.nodeInput[10],metadata.movementLearning.qualifiedWeights.nodeInput[10]);Assert.IsTrue(metadata.movementLearning.simpleBodyQualified);Assert.AreEqual(1,metadata.sharedLearning.m3.usefulReplay.Count);Assert.AreEqual(hash,NativeTensorCheckpointStore.Sha256(file));using(var stream=new FileStream(file,FileMode.Open,FileAccess.Write))stream.SetLength(stream.Length/2);Assert.Catch<Exception>(()=>NativeTensorCheckpointStore.Read(file,JsonConvert.DeserializeObject<NativeEcosystemCheckpoint>(json)));}finally{Directory.Delete(directory,true);}
    }

    [Test]
    public void V26SchemaRejectsOlderFingerprints()
    {
        var checkpoint=new NativeEcosystemCheckpoint();Assert.AreEqual(26,checkpoint.schema);StringAssert.Contains("native-v26",checkpoint.configuration);Assert.IsFalse(NativeEcosystemCheckpoint.IsCompatibleConfiguration("native-v25"));Assert.IsFalse(NativeEcosystemCheckpoint.IsCompatibleConfiguration("native-v20"));
    }

    [Test]
    public void PopulationTopologyUsesFourMatchingIslandsAndSlots()
    {
        Assert.AreEqual(4,NativeEcosystemController.InitialBodySpeciesCount);Assert.AreEqual(4,WorldLayout.MaximumNativeSpawnSlots);Assert.AreEqual(4,new CreatureSpawner().creaturesToSpawn);
    }

    [Test]
    public void FixedGoalHeadingFacesGoalAndM1GoalRemainsUnprojected()
    {
        var model=new NativeCreatureModel(NativeBrainWeights.Create(50),51);Vector2 start=new Vector2(0f,NativeCreatureModel.FixedGoalWorldHeight);NativeObservation fixedGoal=model.Observe(new float[28],Array.Empty<Limb>(),0,new NativeLocalProprioception{worldPosition=start,worldRotationDegrees=0f},false);Vector2 front=new Vector2(fixedGoal.worldGoalHeading.y,fixedGoal.worldGoalHeading.x);Assert.Greater(Vector2.Dot(front,(fixedGoal.worldGoal-start).normalized),.9999f);var m1=new NativeCreatureModel(NativeBrainWeights.Create(52),53);float[] output={0f,1f,0f,1f};NativeObservation first=m1.Observe(new float[28],Array.Empty<Limb>(),0,new NativeLocalProprioception{worldPosition=new Vector2(10f,3f)},true,output),moved=m1.Observe(new float[28],Array.Empty<Limb>(),20,new NativeLocalProprioception{worldPosition=new Vector2(40f,8f)},true,output);Assert.Greater(first.worldGoal.y,3f);Assert.Greater(moved.worldGoal.y,8f);Assert.AreNotEqual(NativeCreatureModel.FixedGoalWorldHeight,first.worldGoal.y);
    }

    [Test]
    public void BiasExplorationAndTemporaryGoalCadencesRemainStable()
    {
        Assert.AreEqual(.5f,NativeCreatureModel.ActionBiasStrength(0),1e-7f);Assert.Greater(NativeCreatureModel.ActionBiasStrength(16666),0f);Assert.AreEqual(0f,NativeCreatureModel.ActionBiasStrength(16667));Assert.AreEqual(.5f,NativeCreatureModel.ExplorationSigma(0),1e-6f);Assert.AreEqual(.05f,NativeCreatureModel.ExplorationSigma(10000),1e-6f);var unproven=new NativeCreatureModel(NativeBrainWeights.Create(72),73);Assert.AreEqual(NativeCreatureModel.AssistedBiasStrength,unproven.EffectiveActionBias(50000));Assert.AreEqual(.15f,unproven.EffectiveExplorationSigma(50000));Assert.AreEqual(-NativeCreatureModel.BiasAction(0f,NativeCreatureModel.AssistedBiasStrength,0,25,1),NativeCreatureModel.BiasAction(0f,NativeCreatureModel.AssistedBiasStrength,0,25,-1),1e-7f);Assert.Greater(Mathf.Abs(NativeCreatureModel.BiasAction(0f,NativeCreatureModel.AssistedBiasStrength,0,25,1)),7f);Assert.AreEqual(12f,NativeCreatureModel.FixedTrainingGoal(999).x);Assert.AreEqual(-12f,NativeCreatureModel.FixedTrainingGoal(1000).x);
    }

    [Test]
    public void ExistingVisualAndDeathContractsRemainIntact()
    {
        Assert.AreEqual(50,NativeEnergyBar.RefreshIntervalTicks);Assert.AreEqual(100f,NativeEcosystemController.TemporaryGoalRewardEnergy);Assert.AreEqual(1.25f,NativeEcosystemController.TemporaryGoalReachRadius);GameObject body=new GameObject("death-gate-test");try{CreatureBrain brain=body.AddComponent<CreatureBrain>();brain.DeathsEnabled=false;brain.Kill("blocked");Assert.IsFalse(brain.IsDead);}finally{UnityEngine.Object.DestroyImmediate(body);}
    }

    [Test]
    public void RtNeatUsesTwentyEightInputsFourOutputsAndStructuralInnovations()
    {
        var policy=new NativeRtNeatGoalPolicy();RtNeatGoalGenome genome=policy.Seed();Assert.AreEqual(33,genome.nodes.Count);Assert.AreEqual(29*4,genome.connections.Count);int innovations=policy.innovations.Count;policy.Split(genome);Assert.Greater(genome.nodes.Count,33);Assert.Greater(policy.innovations.Count,innovations);var phenotype=new RtNeatGoalPhenotype(genome);float[] output=new float[4];phenotype.Evaluate(new float[28],output);Assert.AreEqual(4,output.Length);Assert.IsTrue(output.All(NativeCreatureModel.IsFinite));
    }

    [Test]
    public void ReservoirIsBoundedAndSamplesWithoutTaskLabels()
    {
        var replay=new NativeReplayBuffer();for(int i=0;i<NativeReplayBuffer.Capacity+257;i++){NativeCompletedSequence sample=SequenceFixture(i.ToString());sample.bodySignature=i%2==0?"body-a":"body-b";sample.controllerId="controller-"+i;replay.Offer(sample);}Assert.AreEqual(NativeReplayBuffer.Capacity,replay.Count);Assert.AreEqual(NativeReplayBuffer.Capacity+257,replay.seen);Assert.AreEqual(3,replay.Sample(3).Count);Assert.IsTrue(replay.Sample(16,"body-a").All(s=>s.bodySignature=="body-a"));
    }

    [Test]
    public void SynapticIntelligenceConsolidatesAndPenalizesDriftWithoutBoundaries()
    {
        float[] parameter={1f};var parameters=new List<KeyValuePair<string,float[]>>{new KeyValuePair<string,float[]>("p",parameter)};var gradients=new Dictionary<string,float[]>{{"p",new[]{1f}}};var si=new NativeSynapticIntelligence();for(int step=1;step<=NativeSynapticIntelligence.WarmupUpdates;step++){si.BeginStep(parameters,gradients);parameter[0]-=.001f;si.EndStep(parameters,step);}NativeSynapticSlot slot=si.Get("p",1,parameter);Assert.Greater(slot.importance[0],0f);parameter[0]+=.5f;var penalized=new Dictionary<string,float[]>{{"p",new[]{0f}}};si.AddPenalty(parameters,penalized);Assert.Greater(penalized["p"][0],0f);Assert.LessOrEqual(slot.importance[0],NativeSynapticIntelligence.ImportanceCap);
    }

    [Test]
    public void CurriculumStagesQualificationPracticeAndRetentionAreIndependent()
    {
        CollectionAssert.AreEqual(new[]{0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,5f,5f,10f,10f,15f,15f},Enumerable.Range(0,NativeSuccessCurriculum.StageCount).Select(NativeSuccessCurriculum.SlopeDegrees).ToArray());CollectionAssert.AreEqual(new[]{1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1},Enumerable.Range(0,NativeSuccessCurriculum.StageCount).Select(NativeSuccessCurriculum.Direction).ToArray());CollectionAssert.AreEqual(new[]{.5f,.5f,1f,1f,2f,2f,5f,5f,40f,40f,50f,50f,60f,60f,60f,60f},Enumerable.Range(0,NativeSuccessCurriculum.StageCount).Select(NativeSuccessCurriculum.Distance).ToArray());Assert.AreEqual(.1f,NativeSuccessCurriculum.GoalTolerance(0),1e-6f);Assert.AreEqual(.25f,NativeSuccessCurriculum.GoalTolerance(6),1e-6f);Assert.AreEqual(.5f,NativeSuccessCurriculum.GoalTolerance(8),1e-6f);
        var state=new NativeCurriculumState{mode=NativeTrialMode.Training};NativeBrainWeights weights=NativeBrainWeights.Create(91);for(int i=0;i<10;i++)NativeSuccessCurriculum.RegisterSuccess(state,weights);Assert.AreEqual(NativeTrialMode.Qualification,state.mode);Assert.IsNotNull(state.qualificationSnapshot);for(int i=0;i<10;i++)NativeSuccessCurriculum.RegisterSuccess(state,weights);NativeSuccessCurriculum.RegisterFailure(state,weights);NativeSuccessCurriculum.RegisterFailure(state,weights);Assert.AreEqual(1,state.stage);CollectionAssert.AreEqual(new[]{0},state.passedStages);Assert.AreEqual(NativeTrialMode.Training,state.mode);
        state.ordinaryTrials=4;NativeSuccessCurriculum.StartNextTrial(state);Assert.AreEqual(NativeTrialMode.Practice,state.mode);var rng=new NativeDeterministicRng(7);Assert.AreEqual(0,NativeSuccessCurriculum.TrialStage(state,ref rng));state.ordinaryTrials=19;state.mode=NativeTrialMode.Settling;NativeSuccessCurriculum.StartNextTrial(state);Assert.AreEqual(NativeTrialMode.Retention,state.mode);state.stage=NativeSuccessCurriculum.StageCount;state.mode=NativeTrialMode.Settling;NativeSuccessCurriculum.StartNextTrial(state);Assert.AreEqual(NativeTrialMode.Practice,state.mode);
    }

    [Test]
    public void CurriculumEndpointsKeepRightAndLeftTrialsOnOppositeIslandSides()
    {
        Vector2 center=new Vector2(48f,-4.5f);NativeEcosystemController.CurriculumEndpointsForTesting(center,0f,78f,1,68f,0f,out Vector2 rightSpawn,out Vector2 rightGoal);Assert.AreEqual(14f,rightSpawn.x,1e-6f);Assert.AreEqual(82f,rightGoal.x,1e-6f);NativeEcosystemController.CurriculumEndpointsForTesting(center,0f,78f,-1,68f,0f,out Vector2 leftSpawn,out Vector2 leftGoal);Assert.AreEqual(82f,leftSpawn.x,1e-6f);Assert.AreEqual(14f,leftGoal.x,1e-6f);
        NativeEcosystemController.CurriculumEndpointsForTesting(center,15f,78f,1,68f,0f,out Vector2 slopedSpawn,out Vector2 slopedGoal);Assert.Greater(Vector2.Dot(slopedGoal-slopedSpawn,new Vector2(Mathf.Cos(15f*Mathf.Deg2Rad),Mathf.Sin(15f*Mathf.Deg2Rad))),0f);
    }

    [Test]
    public void ExpandedWorldUsesWideEcosystemSpawnGrid()
    {
        Assert.AreEqual(480f,WorldLayout.WorldBoundaryHalfWidth);Assert.AreEqual(472f,WorldLayout.GoalHalfWidth);Assert.AreEqual(new Vector2(-320f,WorldLayout.GroundTop+2.5f),WorldLayout.CreatureSpawnPosition(0));Assert.AreEqual(new Vector2(320f,WorldLayout.GroundTop+2.5f),WorldLayout.CreatureSpawnPosition(4));
    }

    [Test]
    public void QualificationFailureRequiresFiveFreshTrainingSuccessesBeforeRetry()
    {
        var state=new NativeCurriculumState{mode=NativeTrialMode.Training};NativeBrainWeights weights=NativeBrainWeights.Create(92);for(int i=0;i<10;i++)NativeSuccessCurriculum.RegisterSuccess(state,weights);for(int i=0;i<12;i++)NativeSuccessCurriculum.RegisterFailure(state,weights);Assert.AreEqual(NativeTrialMode.Training,state.mode);Assert.AreEqual(0,state.retrySuccesses);for(int i=0;i<4;i++)NativeSuccessCurriculum.RegisterSuccess(state,weights);Assert.AreEqual(NativeTrialMode.Training,state.mode);NativeSuccessCurriculum.RegisterSuccess(state,weights);Assert.AreEqual(NativeTrialMode.Qualification,state.mode);
    }

    [Test]
    public void V26GraphPolicyHasSharedEncoderAndTwoControlTimescales()
    {
        NativeMovementWeights weights=NativeMovementWeights.Create(26001);Assert.AreEqual(NativeMovementWeights.NodeInput*64,weights.nodeInput.Length);Assert.AreEqual(64*64,weights.messageParent.Length);Assert.AreEqual(NativeMovementWeights.M3Input*NativeMovementWeights.TargetSize,weights.m3Head0.Length);Assert.AreEqual(weights.m3Head0.Length,weights.m3Head1.Length);Assert.AreEqual(weights.m3Head0.Length,weights.m3Head2.Length);using(var learner=new NativeMovementLearner(new NativeMovementLearningState{weights=weights})){NativeMovementObservation first=MovementObservationFixture(10,0f);NativePolicyDecision slow=learner.Decide(first,true,false);NativeMovementObservation next=MovementObservationFixture(15,100f);next.heldRhythm=(float[])slow.heldRhythm.Clone();NativePolicyDecision fast=learner.Decide(next,false,false);Assert.IsTrue(slow.sampledRhythm);Assert.IsFalse(fast.sampledRhythm);CollectionAssert.AreEqual(slow.heldRhythm,fast.heldRhythm);Assert.That(fast.frequency,Is.InRange(.25f,3f));Assert.IsTrue(fast.requested.All(x=>x>=-8f&&x<=8f));}
    }

    [Test]
    public void V26PolicyIgnoresWorldTranslationAndUsesGraphGradient()
    {
        NativeMovementWeights weights=NativeMovementWeights.Create(26002);NativeMovementObservation a=MovementObservationFixture(10,0f),b=MovementObservationFixture(10,200f);CollectionAssert.AreEqual(NativeMovementLearner.PolicyMeanForTesting(a,weights),NativeMovementLearner.PolicyMeanForTesting(b,weights));float[] analytic=NativeMovementLearner.PolicyGradientForTesting(a,weights,"nodeInput",1);int index=Largest(analytic);float original=weights.nodeInput[index],epsilon=1e-3f;weights.nodeInput[index]=original+epsilon;float plus=NativeMovementLearner.PolicyMeanForTesting(a,weights)[1];weights.nodeInput[index]=original-epsilon;float minus=NativeMovementLearner.PolicyMeanForTesting(a,weights)[1];AssertRelative(analytic[index],(plus-minus)/(2f*epsilon),.02f);
    }

    [Test]
    public void V26RewardPpoAndAuxiliaryGradientsAreNumericallyConsistent()
    {
        Assert.AreEqual(.1f,NativeMovementLearner.TravelReward(new Vector2(1f,0f),Vector2.right,0f,0f),1e-7f);Assert.Less(NativeMovementLearner.TravelReward(new Vector2(-.05f,0f),Vector2.right,0f,0f),0f);Assert.Greater(NativeMovementLearner.TravelReward(Vector2.zero,Vector2.zero,1f,.2f),0f);Assert.Greater(NativeMovementLearner.EffortPenalty(new[]{8f,0f},2),0f);Assert.AreEqual(-1.2f,NativeMovementLearner.PpoSurrogate(2f,1f),1e-6f);float sample=.7f,mean=.2f,std=.8f,epsilon=1e-4f;float numerical=(NativeMovementLearner.GaussianLogProbability(sample,mean+epsilon,std)-NativeMovementLearner.GaussianLogProbability(sample,mean-epsilon,std))/(2f*epsilon);Assert.AreEqual(NativeMovementLearner.GaussianMeanGradient(sample,mean,std),numerical,2e-3f);float score=.4f;float inverseNumerical=(NativeMovementLearner.InverseLoss(score+epsilon,1f)-NativeMovementLearner.InverseLoss(score-epsilon,1f))/(2f*epsilon);Assert.AreEqual(NativeMovementLearner.InverseGradient(score,1f),inverseNumerical,2e-3f);float value=.3f,target=.9f;float valueNumerical=(NativeMovementLearner.ValueLoss(value+epsilon,target)-NativeMovementLearner.ValueLoss(value-epsilon,target))/(2f*epsilon);Assert.AreEqual(value-target,valueNumerical,2e-3f);
    }

    [Test]
    public void V26GeometryAndHorizonContractsRemainCausal()
    {
        Vector2 b=new Vector2(.2f,.1f),e=new Vector2(1.2f,.1f),parent=new Vector2(.05f,-.02f),target=new Vector2(.7f,.8f);float angle=.3f,epsilon=1e-4f;float numerical=(NativeMovementLearner.GeometryEndpointLoss(b,e,angle+epsilon,parent,target)-NativeMovementLearner.GeometryEndpointLoss(b,e,angle-epsilon,parent,target))/(2f*epsilon);Assert.AreEqual(NativeMovementLearner.GeometryEndpointAngleGradient(b,e,angle,parent,target),numerical,2e-3f);float[] actions={1f,2f,3f,4f,5f,6f},h1=NativeMovementLearner.HorizonEncodingForTesting(actions,6,1),h5=NativeMovementLearner.HorizonEncodingForTesting(actions,6,5),h10=NativeMovementLearner.HorizonEncodingForTesting(actions,6,10);CollectionAssert.AreEqual(actions.Select(x=>x/8f).ToArray(),h1.Take(6).ToArray());CollectionAssert.AreEqual(h1.Take(6),h5.Take(6));CollectionAssert.AreEqual(h5.Take(6),h10.Take(6));CollectionAssert.AreEqual(new[]{1f,0f,0f},h1.Skip(6));CollectionAssert.AreEqual(new[]{0f,1f,0f},h5.Skip(6));CollectionAssert.AreEqual(new[]{0f,0f,1f},h10.Skip(6));
    }

    [Test]
    public void SustainedMovementQualificationUsesSpeedWindowsAndStationaryLimit()
    {
        Assert.AreEqual(8,NativeCurriculumState.QualificationTarget);Assert.AreEqual(10,NativeCurriculumState.QualificationTrials);
        var pass=new NativeSustainedMovementTrial();pass.Reset(0f,1f);float progress=0f;for(int i=0;i<NativeSustainedMovementTrial.RequiredTicks;i++){progress+=.003f;pass.Observe(progress,.02f);}Assert.IsTrue(pass.Qualified(.02f));Assert.GreaterOrEqual(pass.MeanCommandedSpeed(.02f),.1f);Assert.GreaterOrEqual(pass.PositiveWindowFraction,.8f);
        var stalled=new NativeSustainedMovementTrial();stalled.Reset(0f,1f);progress=0f;for(int i=0;i<NativeSustainedMovementTrial.RequiredTicks;i++){if(i<1000||i>1300)progress+=.003f;stalled.Observe(progress,.02f);}Assert.IsFalse(stalled.Qualified(.02f));Assert.Greater(stalled.maxStationaryTicks,NativeSustainedMovementTrial.MaximumStationaryTicks);
        var reversing=new NativeSustainedMovementTrial();reversing.Reset(0f,1f);progress=0f;for(int i=0;i<NativeSustainedMovementTrial.RequiredTicks;i++){progress+=(i/NativeSustainedMovementTrial.WindowTicks)%5==0?-.003f:.003f;reversing.Observe(progress,.02f);}Assert.Less(reversing.PositiveWindowFraction,1f);
    }

    [Test]
    public void QualifiedMovementSnapshotCanRollbackWeightsAndOptimizer()
    {
        NativeMovementWeights current=NativeMovementWeights.Create(91),qualified=NativeMovementWeights.Create(92);var state=new NativeMovementLearningState{weights=current,optimizer=new NativeOptimizerState(),qualifiedWeights=qualified,qualifiedOptimizer=new NativeOptimizerState(),modelVersion=4};state.qualifiedOptimizer.Get("nodeInput",qualified.nodeInput.Length).first[0]=.75f;using(var learner=new NativeMovementLearner(state)){learner.RollbackToQualified();NativeMovementLearningState restored=learner.Capture();Assert.AreEqual(qualified.nodeInput[0],restored.weights.nodeInput[0]);Assert.AreEqual(.75f,restored.optimizer.Get("nodeInput",qualified.nodeInput.Length).first[0]);Assert.AreEqual(5,restored.modelVersion);}
    }

    private static NativeMovementObservation MovementObservationFixture(int tick,float worldX)
    {
        NativeObservation current=ObservationFixture(tick);current.torsoWithoutGoal[0]=worldX;current.torsoWithGoal[0]=worldX;return new NativeMovementObservation{history=Enumerable.Range(0,NativeMovementWeights.HistoryLength).Select(_=>NativeReplayBuffer.CloneObservation(current)).ToArray(),oscillatorSin=0f,oscillatorCos=1f,angularVelocity=0f,rhythmTicksRemaining=0f,heldRhythm=new float[1+NativeCreatureModel.MaxLimbs*3],limbCount=current.limbCount,episode=1,tick=tick,controllerId="test",bodySignature="body",command=Vector2.right,gravityLocal=Vector2.down};
    }

    private static NativeObservation ObservationFixture(int tick)
    {
        var o=new NativeObservation{torsoWithGoal=new float[NativeCreatureModel.M2ContextSize],torsoWithoutGoal=new float[NativeCreatureModel.M3ContextSize],jointFeatures=new float[NativeCreatureModel.MaxLimbs*NativeCreatureModel.JointFeatureSize],positionEncodings=new float[NativeCreatureModel.MaxLimbs*NativeCreatureModel.PositionEncodingSize],limbMask=new float[NativeCreatureModel.MaxLimbs],parentSlots=new int[NativeCreatureModel.MaxLimbs],limbCount=2,goal=new Vector2(.4f,.1f),goalHeading=new Vector2(.2f,.98f),tick=tick};o.limbMask[0]=o.limbMask[1]=1f;Array.Copy(new[]{.1f,.05f,.2f,-.1f,.3f,.95f,.4f,.1f,.2f,.98f},o.torsoWithGoal,10);Array.Copy(new[]{.1f,.05f,.2f,-.1f,.3f,.95f},o.torsoWithoutGoal,6);for(int probe=0;probe<3;probe++){int a=10+probe*4,b=6+probe*4;o.torsoWithGoal[a]=o.torsoWithoutGoal[b]=1f;o.torsoWithGoal[a+1]=o.torsoWithoutGoal[b+1]=.25f;o.torsoWithGoal[a+3]=o.torsoWithoutGoal[b+3]=1f;}Array.Copy(NativeCreatureModel.JointFeatures(15f,30f,true,false,false),0,o.jointFeatures,0,NativeCreatureModel.JointFeatureSize);Array.Copy(NativeCreatureModel.JointFeatures(-20f,-40f,false,true,true),0,o.jointFeatures,NativeCreatureModel.JointFeatureSize,NativeCreatureModel.JointFeatureSize);Array.Copy(NativeCreatureModel.JointPositionEncoding(new[]{1},1),0,o.positionEncodings,0,NativeCreatureModel.PositionEncodingSize);Array.Copy(NativeCreatureModel.JointPositionEncoding(new[]{2,3},3),0,o.positionEncodings,NativeCreatureModel.PositionEncodingSize,NativeCreatureModel.PositionEncodingSize);return o;
    }
    private static NativeCompletedSequence SequenceFixture(string id){float[] actions=new float[NativeCreatureModel.PlanningHorizon*NativeCreatureModel.MaxLimbs];for(int i=0;i<actions.Length;i++)actions[i]=(i%7-3)*.2f;float[] poses=new float[NativeCreatureModel.M3OutputSize],aux=new float[NativeCreatureModel.M3AuxiliaryOutputSize];for(int s=0;s<NativeCreatureModel.PlanningHorizon;s++){int p=s*NativeCreatureModel.PoseValueCount;poses[p]=.02f*(s+1);poses[p+1]=-.005f*s;Vector2 heading=NativeCreatureModel.HeadingEncoding(s*5f);poses[p+2]=heading.x;poses[p+3]=heading.y;int a=s*NativeCreatureModel.AuxiliaryValueCount;aux[a]=.01f*s;aux[a+2]=s%2;aux[a+3]=.25f;aux[a+4]=.5f;}return new NativeCompletedSequence{sampleId=id,observation=ObservationFixture(10000),actions=actions,actualPoses=poses,actualAuxiliary=aux};}
    private static NativeCompletedSequence CloneSequence(NativeCompletedSequence sample)=>JsonConvert.DeserializeObject<NativeCompletedSequence>(JsonConvert.SerializeObject(sample));
    private static NativeEcosystemCheckpoint CheckpointFixture(){var policy=NativeBrainWeights.Create(60);var dynamics=NativeDynamicsWeights.Create(61);var model=new NativeCreatureModel(policy,62);model.OptimizeAndPlan(ObservationFixture(10000),dynamics,0);NativeCreatureSave snapshot=model.Capture();NativeCompletedSequence replay=SequenceFixture("sample");replay.bodySignature="body-a";replay.controllerId="creature-a";var curriculum=new NativeCurriculumState{qualificationLearningState=snapshot.Clone(),validatedLearningState=snapshot.Clone(),qualificationSnapshot=snapshot.weights.Clone(),validatedSnapshot=snapshot.weights.Clone()};var m3=new NativeM3LearningState{dynamics=dynamics};m3.replay.Offer(replay);m3.validation.Add(replay.Clone());return new NativeEcosystemCheckpoint{rtNeat=new NativeRtNeatGoalPolicy(),creatures=new List<NativeCreatureCheckpoint>{new NativeCreatureCheckpoint{creatureId="creature-a",brain=model.Capture(),m3=m3,m1Genome=new NativeRtNeatGoalPolicy().Seed(),curriculum=curriculum}}};}
    private static int Largest(float[] values){int index=0;for(int i=1;i<values.Length;i++)if(Mathf.Abs(values[i])>Mathf.Abs(values[index]))index=i;Assert.Greater(Mathf.Abs(values[index]),1e-7f);return index;}
    private static float Difference(float[] a,float[] b){float sum=0f;for(int i=0;i<Math.Min(a.Length,b.Length);i++)sum+=Mathf.Abs(a[i]-b[i]);return sum;}
    private static float Dot(float[] a,float[] b){float sum=0f;for(int i=0;i<a.Length;i++)sum+=a[i]*b[i];return sum;}
    private static void AssertRelative(float analytic,float numerical,float tolerance){float relative=Mathf.Abs(analytic-numerical)/Mathf.Max(1e-5f,Mathf.Abs(analytic)+Mathf.Abs(numerical));Assert.Less(relative,tolerance,$"analytic={analytic} numerical={numerical}");}
}
