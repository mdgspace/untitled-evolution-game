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
    public void V18WidthsDepthsAndHeadPartitionsAreExact()
    {
        Assert.AreEqual(32,NativeCreatureModel.M1Width);Assert.AreEqual(128,NativeCreatureModel.M2Width);Assert.AreEqual(128,NativeCreatureModel.M3Width);
        Assert.AreEqual(2,NativeCreatureModel.M2TransformerBlocks);Assert.AreEqual(3,NativeCreatureModel.M3TransformerBlocks);Assert.AreEqual(4,NativeCreatureModel.M2AttentionHeads);Assert.AreEqual(4,NativeCreatureModel.M3AttentionHeads);Assert.AreEqual(32,NativeCreatureModel.M2Width/NativeCreatureModel.M2AttentionHeads);
        Assert.AreEqual(3,NativeCreatureModel.M2MlpHiddenLayers);Assert.AreEqual(4,NativeCreatureModel.M3MlpHiddenLayers);
        NativeBrainWeights policy=NativeBrainWeights.Create(1);NativeDynamicsWeights dynamics=NativeDynamicsWeights.Create(2);
        Assert.AreEqual(28*32,policy.m1Input.Length);Assert.AreEqual(10*128,policy.m2Context.Length);Assert.AreEqual(7*128,policy.m2Joint.Length);Assert.AreEqual(16*128,policy.m2Position.Length);Assert.AreEqual(5*128,policy.m2StepEmbedding.Length);Assert.AreEqual(128,policy.m2Output.Length);
        Assert.AreEqual(NativeCreatureModel.M2ParameterCount,policy.M2Parameters().Sum(p=>p.Value.Length));
        Assert.AreEqual(12*128,dynamics.joint.Length);Assert.AreEqual(5*128,dynamics.horizonQueries.Length);Assert.AreEqual(128*9,dynamics.output.Length);
    }

    [Test]
    public void PolicyOwnsOnlyM1M2AndDynamicsIsSingleSeparateType()
    {
        string policyFields=string.Join(" ",typeof(NativeBrainWeights).GetFields().Select(f=>f.Name)).ToLowerInvariant();StringAssert.DoesNotContain("m3",policyFields);StringAssert.DoesNotContain("recurrent",policyFields);StringAssert.DoesNotContain("gru",policyFields);
        Assert.IsTrue(typeof(NativeEcosystemCheckpoint).GetField("sharedM3").FieldType==typeof(NativeDynamicsWeights));
    }

    [Test]
    public void ContactAndOrientationJointFeaturesAreExact()
    {
        float[] f=NativeCreatureModel.JointFeatures(90f,360f,true,false,true);Assert.AreEqual(7,f.Length);Assert.AreEqual(1f,f[0],1e-5f);Assert.AreEqual(0f,f[1],1e-5f);Assert.AreEqual(0f,f[2],1e-5f);Assert.AreEqual(-.5f,f[3],1e-5f);CollectionAssert.AreEqual(new[]{1f,0f,1f},f.Skip(4).ToArray());
    }

    [Test]
    public void TopologyEncodingIsDeterministicUniformAndIdentityFree()
    {
        float[] a=NativeCreatureModel.JointPositionEncoding(new[]{5,3},99),b=NativeCreatureModel.JointPositionEncoding(new[]{5,3},-7),c=NativeCreatureModel.JointPositionEncoding(new[]{5,2},99);Assert.AreEqual(16,a.Length);CollectionAssert.AreEqual(a,b);Assert.Greater(Difference(a,c),1e-4f);Assert.AreEqual(1f,a[0]);Assert.AreEqual(1f,a[5]);Assert.AreEqual(0f,a[10]);Assert.AreEqual(2f/3f,a[15],1e-6f);
    }

    [Test]
    public void PaddedLimbFeaturesAndActionsAreMasked()
    {
        NativeObservation clean=ObservationFixture(10000),padded=JsonConvert.DeserializeObject<NativeObservation>(JsonConvert.SerializeObject(clean));int limb=NativeCreatureModel.MaxLimbs-1;for(int i=0;i<7;i++)padded.jointFeatures[limb*7+i]=1000+i;for(int i=0;i<16;i++)padded.positionEncodings[limb*16+i]=-1000-i;NativeBrainWeights policy=NativeBrainWeights.Create(3);NativeDynamicsWeights dynamics=NativeDynamicsWeights.Create(4);Assert.AreEqual(NativeCreatureModel.M2LossForTesting(clean,policy,dynamics),NativeCreatureModel.M2LossForTesting(padded,policy,dynamics),1e-6f);NativeCompletedSequence first=SequenceFixture("a"),second=CloneSequence(first);for(int s=0;s<5;s++)second.actions[s*NativeCreatureModel.MaxLimbs+limb]=999f;Assert.AreEqual(NativeCreatureModel.M3LossForTesting(first,dynamics),NativeCreatureModel.M3LossForTesting(second,dynamics),1e-6f);
    }

    [Test]
    public void SiLUAndRmsNormGradientsMatchCentralDifference()
    {
        float x=.37f,epsilon=1e-3f;float numerical=(NativeCreatureModel.SiLU(x+epsilon)-NativeCreatureModel.SiLU(x-epsilon))/(2f*epsilon);Assert.AreEqual(numerical,NativeCreatureModel.SiLUDerivative(x),1e-4f);
        float[] input=Enumerable.Range(0,128).Select(i=>(i-63f)/80f).ToArray(),gamma=Enumerable.Repeat(1f,128).ToArray(),upstream=Enumerable.Range(0,128).Select(i=>Mathf.Sin(i*.13f)).ToArray();float[] analytic=NativeCreatureModel.RmsNormInputGradientForTesting(input,gamma,upstream);int index=37;float original=input[index];input[index]=original+epsilon;float plus=Dot(NativeCreatureModel.RmsNormForTesting(input,gamma),upstream);input[index]=original-epsilon;float minus=Dot(NativeCreatureModel.RmsNormForTesting(input,gamma),upstream);AssertRelative(analytic[index],(plus-minus)/(2f*epsilon),.002f);
    }

    [Test]
    public void M3PoseAndAuxiliaryGradientsReachSharedOutputAndQueries()
    {
        NativeDynamicsWeights dynamics=NativeDynamicsWeights.Create(11);NativeCompletedSequence sample=SequenceFixture("gradient");float[] output=NativeCreatureModel.M3GradientForTesting(sample,dynamics,"output"),queries=NativeCreatureModel.M3GradientForTesting(sample,dynamics,"horizonQueries");Assert.Greater(output.Sum(Mathf.Abs),1e-6f);Assert.Greater(queries.Sum(Mathf.Abs),1e-6f);
        int index=Largest(output);float original=dynamics.output[index],epsilon=5e-4f;dynamics.output[index]=original+epsilon;float plus=NativeCreatureModel.M3LossForTesting(sample,dynamics);dynamics.output[index]=original-epsilon;float minus=NativeCreatureModel.M3LossForTesting(sample,dynamics);AssertRelative(output[index],(plus-minus)/(2f*epsilon),.05f);
    }

    [Test]
    public void M3AuxiliaryLossUsesVelocityMseAndContactBce()
    {
        float[] predicted=new float[25],actual=new float[25];for(int s=0;s<5;s++){predicted[s*5+2]=predicted[s*5+3]=predicted[s*5+4]=.5f;actual[s*5]=.5f;actual[s*5+2]=1f;}float total=NativeCreatureModel.M3AuxiliaryLoss(predicted,actual,null,out float velocity,out float contact);Assert.Greater(velocity,0f);Assert.Greater(contact,0f);Assert.AreEqual(velocity+contact,total,1e-7f);Assert.IsTrue(NativeCreatureModel.IsFinite(total));
    }

    [Test]
    public void BoundaryBatchIsOneOrderIndependentAdamStep()
    {
        var a=SequenceFixture("a");var b=SequenceFixture("b");b.actualPoses[0]+=.3f;NativeDynamicsWeights left=NativeDynamicsWeights.Create(20),right=left.Clone();var lo=new NativeOptimizerState();var ro=new NativeOptimizerState();NativeM3BatchMetrics lm=NativeCreatureModel.TrainSharedM3Batch(new[]{a,b},left,lo),rm=NativeCreatureModel.TrainSharedM3Batch(new[]{b,a},right,ro);Assert.AreEqual(2,lm.sampleCount);Assert.AreEqual(1,lm.optimizerSteps);Assert.AreEqual(1,lo.step);Assert.AreEqual(0f,Difference(left.output,right.output),1e-7f);Assert.AreEqual(lm.loss,rm.loss,1e-7f);
    }

    [Test]
    public void M2GradientFlowsThroughFrozenM3AndSharedStepDecoder()
    {
        NativeBrainWeights policy=NativeBrainWeights.Create(30);NativeDynamicsWeights dynamics=NativeDynamicsWeights.Create(31),before=dynamics.Clone();NativeObservation observation=ObservationFixture(10000);float[] gradient=NativeCreatureModel.M2GradientForTesting(observation,policy,dynamics,"stepEmbedding");Assert.Greater(gradient.Sum(Mathf.Abs),1e-6f);var model=new NativeCreatureModel(policy,32);NativeActionSequence sequence=model.OptimizeAndPlan(observation,dynamics,100);Assert.AreEqual(1,model.M2Optimizer.step);Assert.AreEqual(0f,Difference(before.output,dynamics.output));for(int s=0;s<5;s++)for(int l=0;l<sequence.limbCount;l++)Assert.That(sequence.actions[s,l],Is.InRange(-8f,8f));
    }

    [Test]
    public void MutationScalingCompensatesForWiderPolicy()
    {
        Assert.That(NativeCreatureModel.M2MutationScale,Is.GreaterThan(0f).And.LessThan(1f));Assert.Greater(NativeBrainWeights.Create(40).M2Parameters().Sum(p=>p.Value.Length),NativeCreatureModel.V16M2ParameterCount);
    }

    [Test]
    public void PoseLossWeightsAndCadenceRemainExact()
    {
        Assert.AreEqual(5,NativeCreatureModel.ActionTicks);Assert.AreEqual(25,NativeCreatureModel.SequenceTicks);Assert.AreEqual(25,NativeCreatureModel.M1GoalWindowTicks);Assert.AreEqual(1000,NativeCreatureModel.GoalWindowTicks);CollectionAssert.AreEqual(new[]{1f,1f,2f,4f,8f},NativeCreatureModel.GoalStepWeights);Assert.AreEqual(1.25f,NativeCreatureModel.M2HeadingLossWeight);float[] p=new float[20];for(int s=0;s<5;s++)p[s*4+3]=1f;p[0]=1f;float early=NativeCreatureModel.M2GoalLoss(p,Vector2.zero,Vector2.up);p=new float[20];for(int s=0;s<5;s++)p[s*4+3]=1f;p[16]=1f;Assert.AreEqual(8f*early,NativeCreatureModel.M2GoalLoss(p,Vector2.zero,Vector2.up),1e-6f);
    }

    [Test]
    public void AntiZeroLossPunishesMissedMotionAndIsFiniteNearRest()
    {
        float[] actual=new float[20];for(int s=0;s<5;s++)actual[s*4+3]=1f;actual[16]=.1f;float zero=NativeCreatureModel.M3Loss(new float[20],actual,null,out float mse,out float relative),correct=NativeCreatureModel.M3Loss((float[])actual.Clone(),actual,null,out _,out _);Assert.Greater(relative,mse*2f);Assert.AreEqual(0f,correct,1e-7f);Assert.IsTrue(NativeCreatureModel.IsFinite(zero));
    }

    [Test]
    public void BinaryTensorStoreRoundTripsAndDetectsCorruption()
    {
        string directory=Path.Combine(Path.GetTempPath(),"native-v18-test-"+Guid.NewGuid().ToString("N"));Directory.CreateDirectory(directory);string file=Path.Combine(directory,"native_ecosystem_v18.test.bin");try{var checkpoint=CheckpointFixture();NativeTensorCheckpointStore.Write(file,checkpoint);string hash=NativeTensorCheckpointStore.Sha256(file);string json=JsonConvert.SerializeObject(checkpoint);StringAssert.DoesNotContain("m2Context",json);StringAssert.DoesNotContain("\"slots\"",json);var metadata=JsonConvert.DeserializeObject<NativeEcosystemCheckpoint>(json);NativeTensorCheckpointStore.Read(file,metadata);Assert.AreEqual(checkpoint.sharedM3.output[10],metadata.sharedM3.output[10]);Assert.AreEqual(checkpoint.creatures[0].brain.weights.m2Context[10],metadata.creatures[0].brain.weights.m2Context[10]);Assert.AreEqual(hash,NativeTensorCheckpointStore.Sha256(file));using(var stream=new FileStream(file,FileMode.Open,FileAccess.Write))stream.SetLength(stream.Length/2);Assert.Catch<Exception>(()=>NativeTensorCheckpointStore.Read(file,JsonConvert.DeserializeObject<NativeEcosystemCheckpoint>(json)));}finally{Directory.Delete(directory,true);}
    }

    [Test]
    public void V18SchemaRejectsV17Fingerprint()
    {
        var checkpoint=new NativeEcosystemCheckpoint();Assert.AreEqual(18,checkpoint.schema);StringAssert.Contains("native-v18",checkpoint.configuration);Assert.IsFalse(NativeEcosystemCheckpoint.IsCompatibleConfiguration("native-v17"));
    }

    [Test]
    public void FixedGoalHeadingFacesGoalAndM1GoalRemainsUnprojected()
    {
        var model=new NativeCreatureModel(NativeBrainWeights.Create(50),51);Vector2 start=new Vector2(0f,NativeCreatureModel.FixedGoalWorldHeight);NativeObservation fixedGoal=model.Observe(new float[28],Array.Empty<Limb>(),0,new NativeLocalProprioception{worldPosition=start,worldRotationDegrees=0f},false);Vector2 front=new Vector2(fixedGoal.worldGoalHeading.y,fixedGoal.worldGoalHeading.x);Assert.Greater(Vector2.Dot(front,(fixedGoal.worldGoal-start).normalized),.9999f);var m1=new NativeCreatureModel(NativeBrainWeights.Create(52),53);NativeObservation first=m1.Observe(new float[28],Array.Empty<Limb>(),0,new NativeLocalProprioception{worldPosition=new Vector2(10f,3f)},true),moved=m1.Observe(new float[28],Array.Empty<Limb>(),20,new NativeLocalProprioception{worldPosition=new Vector2(40f,8f)},true);Assert.AreEqual(first.worldGoal,moved.worldGoal);Assert.AreNotEqual(NativeCreatureModel.FixedGoalWorldHeight,first.worldGoal.y);
    }

    [Test]
    public void BiasExplorationAndTemporaryGoalCadencesRemainStable()
    {
        Assert.AreEqual(.5f,NativeCreatureModel.ActionBiasStrength(0),1e-7f);Assert.Greater(NativeCreatureModel.ActionBiasStrength(16666),0f);Assert.AreEqual(0f,NativeCreatureModel.ActionBiasStrength(16667));Assert.AreEqual(.5f,NativeCreatureModel.ExplorationSigma(0),1e-6f);Assert.AreEqual(.05f,NativeCreatureModel.ExplorationSigma(10000),1e-6f);Assert.AreEqual(12f,NativeCreatureModel.FixedTrainingGoal(999).x);Assert.AreEqual(-12f,NativeCreatureModel.FixedTrainingGoal(1000).x);
    }

    [Test]
    public void ExistingVisualAndDeathContractsRemainIntact()
    {
        Assert.AreEqual(50,NativeEnergyBar.RefreshIntervalTicks);Assert.AreEqual(100f,NativeEcosystemController.TemporaryGoalRewardEnergy);Assert.AreEqual(1.25f,NativeEcosystemController.TemporaryGoalReachRadius);GameObject body=new GameObject("death-gate-test");try{CreatureBrain brain=body.AddComponent<CreatureBrain>();brain.DeathsEnabled=false;brain.Kill("blocked");Assert.IsFalse(brain.IsDead);}finally{UnityEngine.Object.DestroyImmediate(body);}
    }

    private static NativeObservation ObservationFixture(int tick)
    {
        var o=new NativeObservation{torsoWithGoal=new[]{.1f,.05f,.2f,-.1f,.3f,.95f,.4f,.1f,.2f,.98f},torsoWithoutGoal=new[]{.1f,.05f,.2f,-.1f,.3f,.95f},jointFeatures=new float[NativeCreatureModel.MaxLimbs*7],positionEncodings=new float[NativeCreatureModel.MaxLimbs*16],limbCount=2,goal=new Vector2(.4f,.1f),goalHeading=new Vector2(.2f,.98f),tick=tick};Array.Copy(NativeCreatureModel.JointFeatures(15f,30f,true,false,false),0,o.jointFeatures,0,7);Array.Copy(NativeCreatureModel.JointFeatures(-20f,-40f,false,true,true),0,o.jointFeatures,7,7);Array.Copy(NativeCreatureModel.JointPositionEncoding(new[]{1},1),0,o.positionEncodings,0,16);Array.Copy(NativeCreatureModel.JointPositionEncoding(new[]{2,3},3),0,o.positionEncodings,16,16);return o;
    }
    private static NativeCompletedSequence SequenceFixture(string id){float[] actions=new float[5*NativeCreatureModel.MaxLimbs];for(int i=0;i<actions.Length;i++)actions[i]=(i%7-3)*.2f;float[] poses=new float[20],aux=new float[25];for(int s=0;s<5;s++){int p=s*4;poses[p]=.02f*(s+1);poses[p+1]=-.005f*s;Vector2 heading=NativeCreatureModel.HeadingEncoding(s*5f);poses[p+2]=heading.x;poses[p+3]=heading.y;int a=s*5;aux[a]=.01f*s;aux[a+2]=s%2;aux[a+3]=.25f;aux[a+4]=.5f;}return new NativeCompletedSequence{sampleId=id,observation=ObservationFixture(10000),actions=actions,actualPoses=poses,actualAuxiliary=aux};}
    private static NativeCompletedSequence CloneSequence(NativeCompletedSequence sample)=>JsonConvert.DeserializeObject<NativeCompletedSequence>(JsonConvert.SerializeObject(sample));
    private static NativeEcosystemCheckpoint CheckpointFixture(){var policy=NativeBrainWeights.Create(60);var dynamics=NativeDynamicsWeights.Create(61);var model=new NativeCreatureModel(policy,62);model.OptimizeAndPlan(ObservationFixture(10000),dynamics,0);return new NativeEcosystemCheckpoint{sharedM3=dynamics,m3Optimizer=new NativeOptimizerState(),creatures=new List<NativeCreatureCheckpoint>{new NativeCreatureCheckpoint{creatureId="creature-a",brain=model.Capture()}},validation=new List<NativeCompletedSequence>{SequenceFixture("sample")}};}
    private static int Largest(float[] values){int index=0;for(int i=1;i<values.Length;i++)if(Mathf.Abs(values[i])>Mathf.Abs(values[index]))index=i;Assert.Greater(Mathf.Abs(values[index]),1e-7f);return index;}
    private static float Difference(float[] a,float[] b){float sum=0f;for(int i=0;i<Math.Min(a.Length,b.Length);i++)sum+=Mathf.Abs(a[i]-b[i]);return sum;}
    private static float Dot(float[] a,float[] b){float sum=0f;for(int i=0;i<a.Length;i++)sum+=a[i]*b[i];return sum;}
    private static void AssertRelative(float analytic,float numerical,float tolerance){float relative=Mathf.Abs(analytic-numerical)/Mathf.Max(1e-5f,Mathf.Abs(analytic)+Mathf.Abs(numerical));Assert.Less(relative,tolerance,$"analytic={analytic} numerical={numerical}");}
}
