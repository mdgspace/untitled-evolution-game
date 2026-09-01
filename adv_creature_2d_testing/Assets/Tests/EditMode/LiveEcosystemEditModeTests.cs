using System;
using System.Linq;
using System.Reflection;
using Newtonsoft.Json;
using NUnit.Framework;
using UnityEngine;

public class LiveEcosystemEditModeTests
{
    [Test]
    public void FeedForwardSchemaHasExactOpenLoopShapes()
    {
        NativeBrainWeights w=NativeBrainWeights.Create(1);
        Assert.AreEqual(32,NativeCreatureModel.Hidden);Assert.AreEqual(4,NativeCreatureModel.M2AttentionHeads);Assert.AreEqual(8,NativeCreatureModel.M3AttentionHeads);Assert.AreEqual(5,NativeCreatureModel.M2MlpHiddenLayers);Assert.AreEqual(5,NativeCreatureModel.M3MlpHiddenLayers);Assert.AreEqual(2,NativeCreatureModel.M2TransformerBlocks);Assert.AreEqual(2,NativeCreatureModel.M3TransformerBlocks);Assert.AreEqual(5,NativeCreatureModel.PlanningHorizon);Assert.AreEqual(4,NativeCreatureModel.ActionTicks);Assert.AreEqual(20,NativeCreatureModel.SequenceTicks);
        Assert.AreEqual(28,NativeCreatureModel.M1InputSize);
        Assert.AreEqual(8,NativeCreatureModel.M2ContextSize);Assert.AreEqual(6,NativeCreatureModel.M3ContextSize);Assert.AreEqual(4,NativeCreatureModel.JointFeatureSize);Assert.AreEqual(9,NativeCreatureModel.M3JointInputSize);Assert.AreEqual(10,NativeCreatureModel.M3OutputSize);
        Assert.AreEqual(8*32,w.m2Context.Length);Assert.AreEqual(4*32,w.m2Joint.Length);Assert.AreEqual(32*32,w.m2AttentionQuery2.Length);Assert.AreEqual(32*32,w.m2AttentionKey2.Length);Assert.AreEqual(32*32,w.m2AttentionValue2.Length);Assert.AreEqual(32*32,w.m2Deep1.Length);Assert.AreEqual(32*32,w.m2Deep2.Length);Assert.AreEqual(32*32,w.m2Deep3.Length);Assert.AreEqual(32*5,w.m2Output.Length);Assert.AreEqual(32*7+5,w.m2Bias.Length);Assert.AreEqual(6*32,w.m3Context.Length);Assert.AreEqual(9*32,w.m3Joint.Length);Assert.AreEqual(32*32,w.m3AttentionQuery2.Length);Assert.AreEqual(32*32,w.m3AttentionKey2.Length);Assert.AreEqual(32*32,w.m3AttentionValue2.Length);Assert.AreEqual(32*32,w.m3Deep1.Length);Assert.AreEqual(32*32,w.m3Deep2.Length);Assert.AreEqual(32*32,w.m3Deep3.Length);Assert.AreEqual(32*10,w.m3Output.Length);Assert.AreEqual(32*7+10,w.m3Bias.Length);
        string fields=string.Join(" ",typeof(NativeBrainWeights).GetFields().Select(f=>f.Name)).ToLowerInvariant();StringAssert.DoesNotContain("recurrent",fields);StringAssert.DoesNotContain("history",fields);
    }

    [Test]
    public void PaddedJointTokensAndActionsAreStrictlyMasked()
    {
        NativeBrainWeights m2=NativeBrainWeights.Create(41),m3=NativeBrainWeights.Create(42);NativeObservation clean=ObservationFixture(10000);NativeObservation padded=JsonConvert.DeserializeObject<NativeObservation>(JsonConvert.SerializeObject(clean));
        int paddedJoint=NativeCreatureModel.MaxLimbs-1;for(int i=0;i<4;i++)padded.jointFeatures[paddedJoint*4+i]=1000f+i;for(int i=0;i<2;i++)padded.positionEncodings[paddedJoint*2+i]=-1000f-i;
        Assert.AreEqual(NativeCreatureModel.M2LossForTesting(clean,m2,m3),NativeCreatureModel.M2LossForTesting(padded,m2,m3),1e-7f);
        NativeCompletedSequence a=SequenceFixture(),b=JsonConvert.DeserializeObject<NativeCompletedSequence>(JsonConvert.SerializeObject(a));for(int s=0;s<NativeCreatureModel.PlanningHorizon;s++)b.actions[s*NativeCreatureModel.MaxLimbs+paddedJoint]=1000f;
        Assert.AreEqual(NativeCreatureModel.M3LossForTesting(a,m3),NativeCreatureModel.M3LossForTesting(b,m3),1e-7f);
    }

    [Test]
    public void JointAngleAndVelocityUseTwoDimensionalOrientationGeometry()
    {
        float[] f=NativeCreatureModel.JointFeatures(90f,360f);Assert.AreEqual(1f,f[0],1e-5f);Assert.AreEqual(0f,f[1],1e-5f);Assert.AreEqual(0f,f[2],1e-5f);Assert.AreEqual(-.5f,f[3],1e-5f);
        float[] zero=NativeCreatureModel.JointFeatures(0f,-720f);Assert.AreEqual(-1f,zero[2],1e-5f);Assert.AreEqual(0f,zero[3],1e-5f);
    }

    [Test]
    public void JointPathEncodingIsDeterministicAndTopologySensitive()
    {
        float[] a=NativeCreatureModel.JointPositionEncoding(new[]{1,3},5),b=NativeCreatureModel.JointPositionEncoding(new[]{1,3},5),c=NativeCreatureModel.JointPositionEncoding(new[]{1,4},5);CollectionAssert.AreEqual(a,b);Assert.Greater(Mathf.Abs(a[0]-c[0])+Mathf.Abs(a[1]-c[1]),1e-4f);Assert.AreEqual(1f,a[0]*a[0]+a[1]*a[1],1e-5f);
    }

    [Test]
    public void FixedAndM1CadencesAreSequenceAligned()
    {
        Assert.AreEqual(1320,NativeCreatureModel.GoalWindowTicks);Assert.AreEqual(20,NativeCreatureModel.M1GoalWindowTicks);Assert.AreEqual(12f,NativeCreatureModel.FixedTrainingGoal(1319).x);Assert.AreEqual(-12f,NativeCreatureModel.FixedTrainingGoal(1320).x);Assert.AreEqual(12f,NativeCreatureModel.FixedTrainingGoal(2640).x);
    }

    [Test]
    public void TemporaryGoalRewardIsLargeAndHasAReachRadius()
    {
        Assert.AreEqual(100f,NativeEcosystemController.TemporaryGoalRewardEnergy);Assert.AreEqual(1.25f,NativeEcosystemController.TemporaryGoalReachRadius);Vector2 goal=new Vector2(12f,-4f);Assert.IsTrue(NativeEcosystemController.IsTemporaryGoalReached(goal+Vector2.right*1.25f,goal));Assert.IsFalse(NativeEcosystemController.IsTemporaryGoalReached(goal+Vector2.right*1.251f,goal));
    }

    [Test]
    public void M1WorldGoalStaysAnchoredWithoutGroundProjection()
    {
        var model=new NativeCreatureModel(NativeBrainWeights.Create(77),78);float[] global=new float[NativeCreatureModel.M1InputSize];NativeLocalProprioception firstProprio=new NativeLocalProprioception{worldPosition=new Vector2(10f,3f)},movedProprio=new NativeLocalProprioception{worldPosition=new Vector2(40f,8f)};NativeObservation first=model.Observe(global,Array.Empty<Limb>(),0,firstProprio,true),second=model.Observe(global,Array.Empty<Limb>(),NativeCreatureModel.SequenceTicks,movedProprio,true);Assert.AreEqual(first.worldGoal.x,second.worldGoal.x,1e-6f);Assert.AreEqual(first.worldGoal.y,second.worldGoal.y,1e-6f);Assert.AreNotEqual(NativeCreatureModel.FixedGoalWorldHeight,first.worldGoal.y);
    }

    [Test]
    public void GoalDisplayCueIsBoundedButPreservesDirection()
    {
        Vector2 origin=new Vector2(2f,-1f),far=new Vector2(20f,5f),near=new Vector2(3f,-1f);Vector2 cue=NativeGoalVisualizer.DisplayTarget(origin,far);Assert.AreEqual(NativeGoalVisualizer.MaximumDisplayDistance,(cue-origin).magnitude,1e-6f);Assert.Greater(Vector2.Dot(cue-origin,far-origin),0f);Assert.AreEqual(near,NativeGoalVisualizer.DisplayTarget(origin,near));
    }

    [Test]
    public void GoalProgressColorsAreDistinctAndSemanticallyStable()
    {
        Color toward=NativeGoalVisualizer.MovementColor(NativeGoalMovement.Toward),away=NativeGoalVisualizer.MovementColor(NativeGoalMovement.Away),still=NativeGoalVisualizer.MovementColor(NativeGoalMovement.Still);Assert.Greater(toward.g,toward.r);Assert.Greater(away.r,away.g);Assert.Greater(still.r,still.b);
    }

    [Test]
    public void EnergyBarClampsEnergyAndUsesFiftyTickRefreshCadence()
    {
        Assert.AreEqual(50,NativeEnergyBar.RefreshIntervalTicks);Assert.AreEqual(0f,NativeEnergyBar.EnergyFraction(-1f,200f));Assert.AreEqual(.5f,NativeEnergyBar.EnergyFraction(100f,200f));Assert.AreEqual(1f,NativeEnergyBar.EnergyFraction(300f,200f));Assert.AreEqual(0f,NativeEnergyBar.EnergyFraction(1f,0f));Color empty=NativeEnergyBar.EnergyColor(0f),full=NativeEnergyBar.EnergyColor(1f);Assert.Greater(empty.r,empty.g);Assert.Greater(full.g,full.r);
    }

    [Test]
    public void BackLoadedGoalWeightsAreExact()
    {
        CollectionAssert.AreEqual(new[]{1f,1f,2f,4f,8f},NativeCreatureModel.GoalStepWeights);float[] p=new float[10];p[0]=1f;float early=NativeCreatureModel.M2GoalLoss(p,Vector2.zero);p=new float[10];p[8]=1f;float late=NativeCreatureModel.M2GoalLoss(p,Vector2.zero);Assert.AreEqual(8f*early,late,1e-6f);
    }

    [Test]
    public void AntiZeroLossStronglyPunishesMissedRealMotionAndIsFiniteNearRest()
    {
        float[] actual=new float[10];actual[8]=.1f;float zero=NativeCreatureModel.M3Loss(new float[10],actual,null,out float mse,out float relative);float correct=NativeCreatureModel.M3Loss((float[])actual.Clone(),actual,null,out _,out _);Assert.Greater(relative,mse*10f);Assert.AreEqual(0f,correct,1e-7f);
        actual=new float[10];actual[0]=.005f;float near=NativeCreatureModel.M3Loss(new float[10],actual,null,out _,out float nearRelative);Assert.IsTrue(NativeCreatureModel.IsFinite(near));Assert.AreEqual(0f,nearRelative);
    }

    [Test]
    public void BiasUsesParityAndReachesExactlyZeroAtScaledTickCutoff()
    {
        Assert.AreEqual(.5f,NativeCreatureModel.ActionBiasStrength(0),1e-7f);Assert.That(NativeCreatureModel.ActionBiasStrength(6666),Is.InRange(.2499f,.2501f));Assert.Greater(NativeCreatureModel.ActionBiasStrength(13332),0f);Assert.AreEqual(0f,NativeCreatureModel.ActionBiasStrength(13333));
        Assert.AreEqual(.375f,NativeCreatureModel.BiasAction(0f,.5f,0,0),1e-6f);Assert.AreEqual(-.375f,NativeCreatureModel.BiasAction(0f,.5f,1,0),1e-6f);Assert.AreEqual(0f,NativeCreatureModel.BiasAction(0f,0f,0,0));
    }

    [Test]
    public void ExplorationDecaysExponentiallyToPersistentFloor()
    {
        Assert.AreEqual(.5f,NativeCreatureModel.ExplorationSigma(0),1e-6f);Assert.That(NativeCreatureModel.ExplorationSigma(5000),Is.InRange(.05f,.5f));Assert.AreEqual(.05f,NativeCreatureModel.ExplorationSigma(10000),1e-6f);Assert.AreEqual(.05f,NativeCreatureModel.ExplorationSigma(20000),1e-6f);
    }

    [Test]
    public void M3AnalyticOutputGradientMatchesCentralDifference()
    {
        NativeBrainWeights w=NativeBrainWeights.Create(11);NativeCompletedSequence sample=SequenceFixture();float[] analytic=NativeCreatureModel.M3GradientForTesting(sample,w,"output");int index=Largest(analytic);float original=w.m3Output[index],epsilon=1e-3f;w.m3Output[index]=original+epsilon;float plus=NativeCreatureModel.M3LossForTesting(sample,w);w.m3Output[index]=original-epsilon;float minus=NativeCreatureModel.M3LossForTesting(sample,w);w.m3Output[index]=original;AssertRelative(analytic[index],(plus-minus)/(2f*epsilon),.03f);
    }

    [Test]
    public void M2AnalyticGradientThroughFrozenM3MatchesCentralDifference()
    {
        NativeBrainWeights m2=NativeBrainWeights.Create(21),m3=NativeBrainWeights.Create(22);NativeObservation observation=ObservationFixture(10000);float[] analytic=NativeCreatureModel.M2GradientForTesting(observation,m2,m3,"output");int index=Largest(analytic);float original=m2.m2Output[index],epsilon=1e-3f;m2.m2Output[index]=original+epsilon;float plus=NativeCreatureModel.M2LossForTesting(observation,m2,m3);m2.m2Output[index]=original-epsilon;float minus=NativeCreatureModel.M2LossForTesting(observation,m2,m3);m2.m2Output[index]=original;AssertRelative(analytic[index],(plus-minus)/(2f*epsilon),.04f);
    }

    [Test]
    public void OneM2UpdateChangesEveryM2ParameterAndFreezesSharedM3()
    {
        NativeBrainWeights controller=NativeBrainWeights.Create(31),shared=NativeBrainWeights.Create(32),sharedBefore=shared.Clone();var before=controller.M2Parameters().ToDictionary(p=>p.Key,p=>(float[])p.Value.Clone());var model=new NativeCreatureModel(controller,33);NativeActionSequence sequence=model.OptimizeAndPlan(ObservationFixture(10000),shared,0);
        foreach(var parameter in controller.M2Parameters())Assert.Greater(Difference(before[parameter.Key],parameter.Value),0f,parameter.Key);foreach(var pair in shared.M3Parameters().Zip(sharedBefore.M3Parameters(),(a,b)=>new{a,b}))Assert.AreEqual(0f,Difference(pair.a.Value,pair.b.Value),pair.a.Key);Assert.AreEqual(1,model.M2Optimizer.step);for(int s=0;s<5;s++)for(int l=0;l<sequence.limbCount;l++)Assert.That(sequence.actions[s,l],Is.InRange(-8f,8f));
    }

    [Test]
    public void V13CheckpointUsesPrimitiveActionStorageOnly()
    {
        var checkpoint=new NativeEcosystemCheckpoint();Assert.AreEqual(13,checkpoint.schema);StringAssert.Contains("native-v13",checkpoint.configuration);string json=JsonConvert.SerializeObject(SequenceFixture());StringAssert.Contains("actions",json);StringAssert.DoesNotContain("normalized",json);Assert.IsFalse(typeof(NativeCompletedSequence).GetFields().Any(f=>f.FieldType==typeof(float[,])));Assert.IsFalse(typeof(NativeObservation).GetFields().Any(f=>f.FieldType==typeof(Vector2)));Assert.AreEqual(5,JsonConvert.DeserializeObject<NativeCompletedSequence>(json).ActionMatrix().GetLength(0));
    }

    [Test]
    public void ExplorationAndAdamResumeDeterministicallyFromV13CreatureState()
    {
        NativeBrainWeights shared=NativeBrainWeights.Create(51);var first=new NativeCreatureModel(NativeBrainWeights.Create(52),53);first.OptimizeAndPlan(ObservationFixture(10000),shared,321);
        NativeCreatureSave persisted=JsonConvert.DeserializeObject<NativeCreatureSave>(JsonConvert.SerializeObject(first.Capture()));var resumed=new NativeCreatureModel(persisted.weights,999);resumed.Restore(persisted);
        NativeActionSequence expected=first.OptimizeAndPlan(ObservationFixture(13353),shared,322),actual=resumed.OptimizeAndPlan(ObservationFixture(13353),shared,322);
        for(int s=0;s<NativeCreatureModel.PlanningHorizon;s++)for(int l=0;l<expected.limbCount;l++)Assert.AreEqual(expected.actions[s,l],actual.actions[s,l],1e-7f,$"step={s} limb={l}");Assert.AreEqual(first.M2Optimizer.step,resumed.M2Optimizer.step);
    }

    private static NativeObservation ObservationFixture(int tick)
    {
        var o=new NativeObservation{torsoWithGoal=new[]{.1f,.05f,.2f,-.1f,.3f,.95f,.4f,.1f},torsoWithoutGoal=new[]{.1f,.05f,.2f,-.1f,.3f,.95f},jointFeatures=new float[NativeCreatureModel.MaxLimbs*4],positionEncodings=new float[NativeCreatureModel.MaxLimbs*2],limbCount=2,goal=new Vector2(.4f,.1f),tick=tick};
        Array.Copy(NativeCreatureModel.JointFeatures(15f,30f),0,o.jointFeatures,0,4);Array.Copy(NativeCreatureModel.JointFeatures(-20f,-40f),0,o.jointFeatures,4,4);Array.Copy(NativeCreatureModel.JointPositionEncoding(new[]{1},1),0,o.positionEncodings,0,2);Array.Copy(NativeCreatureModel.JointPositionEncoding(new[]{2},5),0,o.positionEncodings,2,2);return o;
    }
    private static NativeCompletedSequence SequenceFixture(){float[] actions=new float[NativeCreatureModel.PlanningHorizon*NativeCreatureModel.MaxLimbs];for(int i=0;i<actions.Length;i++)actions[i]=(i%7-3)*.2f;float[] actual=new float[10];for(int s=0;s<5;s++){actual[s*2]=.02f*(s+1);actual[s*2+1]=-.005f*s;}return new NativeCompletedSequence{observation=ObservationFixture(10000),actions=actions,actualPositions=actual};}
    private static int Largest(float[] values){int index=0;for(int i=1;i<values.Length;i++)if(Mathf.Abs(values[i])>Mathf.Abs(values[index]))index=i;Assert.Greater(Mathf.Abs(values[index]),1e-7f);return index;}
    private static float Difference(float[] a,float[] b){float sum=0f;for(int i=0;i<Math.Min(a.Length,b.Length);i++)sum+=Mathf.Abs(a[i]-b[i]);return sum;}
    private static void AssertRelative(float analytic,float numerical,float tolerance){float relative=Mathf.Abs(analytic-numerical)/Mathf.Max(1e-5f,Mathf.Abs(analytic)+Mathf.Abs(numerical));Assert.Less(relative,tolerance,$"analytic={analytic} numerical={numerical}");}
}
