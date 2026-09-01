using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Newtonsoft.Json;
using UnityEngine;

[DefaultExecutionOrder(-80)]
[DisallowMultipleComponent]
public sealed class NativeEcosystemController : MonoBehaviour
{
    private const float MinimumSpawnSeparation=64f; public const float TemporaryGoalRewardEnergy=100f,TemporaryGoalReachRadius=1.25f; private const int CreatureLayerBase=8,CreatureLayerCount=10,NativeSpawnSlots=6,ValidationCapacity=64;
    public int populationTarget=6;
    [SerializeField] private bool enableM1=false;
    public int controlIntervalTicks=NativeCreatureModel.ActionTicks;
    public float evolutionIntervalSeconds=75f,checkpointIntervalSeconds=30f;
    public bool allowForcedEvolution=true,loadCheckpointOnStart=true;
    public int bodyMutationCooldownGenerations=12;
    public string checkpointFileName="native_ecosystem_v13.json";
    public string Status{get;private set;}="Starting native ecosystem";
    public int Tick{get;private set;} public int Generation{get;private set;} public int Population=>agents.Count; public int SpeciesCount=>agents.Where(a=>a.identity!=null).Select(a=>a.identity.speciesId).Distinct().Count();
    public float LastM3Loss{get;private set;} public float LastM3Mse{get;private set;} public float LastM3RelativeLoss{get;private set;} public float LastM2Loss{get;private set;} public float LastM2GoalLoss{get;private set;} public float LastM2EnergyLoss{get;private set;}
    public int GoalTicksRemaining{get;private set;} public int SequenceStep{get;private set;} public int SharedM3Updates{get;private set;} public float BiasStrength{get;private set;} public float ExplorationSigma{get;private set;}
    public float PlannerMilliseconds{get;private set;} public float M2TrainingMilliseconds{get;private set;} public int M2TrainingUpdates{get;private set;} public float LastActionMagnitude{get;private set;}
    public float LeftProposalLoss{get;private set;}=float.PositiveInfinity; public float RightProposalLoss{get;private set;}=float.PositiveInfinity; public int LeftProposalSamples{get;private set;} public int RightProposalSamples{get;private set;} public string DirectionState{get;private set;}="neutral";
    public Vector2[] LastPredictedPositions{get;private set;}=new Vector2[5]; public Vector2[] LastActualPositions{get;private set;}=new Vector2[5];
    public float MeanM1EnergyFitness=>agents.Count==0?0f:agents.Average(a=>a.m1EnergyFitness); public float RenderRateHz{get;private set;} public string PhysicsMode=>Physics2D.simulationMode.ToString();
    public string CheckpointStatus{get;private set;}="Not saved"; public string LastCheckpointError{get;private set;}=""; public float SecondsUntilCheckpoint=>Mathf.Max(0f,nextCheckpoint-Time.time); public string LastCheckpointSavedAt{get;private set;}="Never";
    public int ActiveControlCount=>agents.Count(a=>a.sequenceActive); public int DeadCreatureCount=>agents.Count(a=>a.brain==null||a.brain.IsDead); public int SettlingCreatureCount=>agents.Count(a=>a.brain!=null&&!a.brain.IsDead&&!a.ReadyForLearning); public int MovingCreatureCount=>agents.Count(a=>a.isSustainedMoving);
    public float MeanRootSpeed=>agents.Where(a=>a.identity?.torso?.Rigidbody!=null&&a.brain!=null&&!a.brain.IsDead).Select(a=>a.identity.torso.Rigidbody.linearVelocity.magnitude).DefaultIfEmpty().Average(); public float MeanJointSpeed=>agents.Where(a=>a.brain!=null&&!a.brain.IsDead).Select(a=>a.brain.MeanJointSpeed).DefaultIfEmpty().Average();
    public int ControlFailures{get;private set;} public string LastControlError{get;private set;}=""; public int ImmediateReplacementCount{get;private set;} public int LastReplacementTick{get;private set;}=-1; public string ReplacementRetryState{get;private set;}="none"; public float PopulationTurns=>Generation/6f; public string ExperimentId{get;private set;}
    public bool M1Enabled=>enableM1; public string M1ActivationSource{get;private set;}="not initialized";
    public bool PredatorKillsEnabled=>Generation>=NativeCreatureModel.FixedGoalGenerations;
    public bool InteractiveFeaturesEnabled=>enableM1&&environmentSpawner!=null&&environmentSpawner.FeaturesEnabled;

    private sealed class NativeAgent
    {
        public CreatureIdentity identity; public CreatureBrain brain; public BodyGenomeDto body; public NativeCreatureModel model; public Limb[] limbs; public NativeGoalVisualizer goalVisualizer; public NativeEnergyBar energyBar;
        public float deathPenalty,m1EnergyFitness,previousEnergy; public bool deathRecorded,protectedOffspring,hasCompletedInitialSettlement,sequenceActive,isSustainedMoving; public int bodyAgeGenerations,reservedSlot,settledSupportTicks,nextFiniteCheckTick,sequenceStartTick,movingEvidenceTicks,stillEvidenceTicks,motionWindowTicks,temporaryGoalRewardBlock=int.MinValue; public Vector2 motionWindowStartPosition,sequenceStartPosition; public float sequenceStartRotation; public NativeObservation observation; public NativeActionSequence sequence; public readonly float[] actualPositions=new float[10]; public readonly float[] globalInputs=new float[28]; public bool ReadyForLearning=>hasCompletedInitialSettlement;
    }
    private readonly List<NativeAgent> agents=new List<NativeAgent>(); private readonly Dictionary<NativeAgent,int> replacementRetries=new Dictionary<NativeAgent,int>(); private readonly List<NativeCompletedSequence> validation=new List<NativeCompletedSequence>(); private List<NativeBodySpecies> species=new List<NativeBodySpecies>();
    private NativeBrainWeights sharedWeights; private NativeOptimizerState m3Optimizer=new NativeOptimizerState(); private System.Random random; private float nextEvolution,nextCheckpoint,rateWindowStarted; private int framesAtRateWindow,controlStep,specialistSelectionCursor; private string checkpointPath; private Predator cachedPredator; private EnvironmentSpawner environmentSpawner;

    private void Awake()
    {
        enableM1=false;M1ActivationSource="locked at startup";Application.runInBackground=true;Physics2D.simulationMode=SimulationMode2D.FixedUpdate;populationTarget=Mathf.Max(6,Mathf.Clamp(populationTarget,1,10));controlIntervalTicks=NativeCreatureModel.ActionTicks;random=new System.Random(1337);sharedWeights=NativeBrainWeights.Create(1337);ExperimentId="openloop-v13-"+DateTime.UtcNow.ToString("yyyyMMdd-HHmmss");
        string persistent=Application.persistentDataPath;foreach(string oldVersion in new[]{"native_ecosystem_v6.json","native_ecosystem_v7.json","native_ecosystem_v8.json","native_ecosystem_v9.json","native_ecosystem_v10.json","native_ecosystem_v11.json","native_ecosystem_v12.json"})foreach(string suffix in new[]{"",".bak",".tmp"}){string old=Path.Combine(persistent,oldVersion+suffix);try{if(File.Exists(old))File.Delete(old);}catch(Exception e){Debug.LogError("Failed to delete abandoned checkpoint "+old+": "+e.Message);}}
        checkpointPath=Path.Combine(persistent,checkpointFileName);environmentSpawner=GetComponent<EnvironmentSpawner>();if(loadCheckpointOnStart)TryLoad();
    }
    private void Start(){SeedPopulation();RebuildSpeciation();nextEvolution=Time.time+evolutionIntervalSeconds;nextCheckpoint=Time.time+checkpointIntervalSeconds;Status="Native open-loop feed-forward ecosystem running";}
    public void EnableInteractiveFeaturesFromUser(){if(environmentSpawner==null)environmentSpawner=FindAnyObjectByType<EnvironmentSpawner>();if(environmentSpawner!=null)environmentSpawner.EnableFeatures();enableM1=true;M1ActivationSource="telemetry button at tick "+Tick;Status="Food, predators, and M1 enabled";}
    private void Update(){if(rateWindowStarted<=0f){rateWindowStarted=Time.unscaledTime;framesAtRateWindow=Time.frameCount;return;}float elapsed=Time.unscaledTime-rateWindowStarted;if(elapsed>=.5f){RenderRateHz=(Time.frameCount-framesAtRateWindow)/elapsed;framesAtRateWindow=Time.frameCount;rateWindowStarted=Time.unscaledTime;}}

    private void FixedUpdate()
    {
        if(Physics2D.simulationMode!=SimulationMode2D.FixedUpdate)Physics2D.simulationMode=SimulationMode2D.FixedUpdate;Tick++;UpdateSettlingState();AdvanceOpenSequences();
        if(Tick%NativeCreatureModel.SequenceTicks==0){controlStep++;TrainCompletedSequences();PlanReadyAgents();}
        foreach(NativeAgent a in agents.ToArray()){if(a.brain==null||a.identity?.torso==null)continue;a.identity.torso.EnergyPenaltyScale=Generation<NativeCreatureModel.FixedGoalGenerations?.1f:1f;a.brain.FixedStep();}
        UpdateTemporaryGoalRewards();if(Tick%NativeEnergyBar.RefreshIntervalTicks==0)UpdateEnergyBarDisplays();UpdateMotionEvidence();if(Tick%50==0)UpdateGoalProgressDisplays();ReplaceDeadCreaturesImmediately();if(agents.Count>0&&agents.All(a=>a.brain==null||a.brain.IsDead))RecoverFromExtinction();if(allowForcedEvolution&&Time.time>=nextEvolution){nextEvolution+=evolutionIntervalSeconds;Evolve();}MaintainPopulationInvariant();
    }

    private void AdvanceOpenSequences()
    {
        SequenceStep=0;
        foreach(NativeAgent a in agents)
        {
            if(!a.sequenceActive||a.brain==null||a.brain.IsDead||a.identity?.torso==null)continue;int elapsed=Tick-a.sequenceStartTick;if(elapsed<=0||elapsed%NativeCreatureModel.ActionTicks!=0)continue;int reached=elapsed/NativeCreatureModel.ActionTicks-1;if(reached<0||reached>=5)continue;
            Vector2 worldDelta=(Vector2)a.identity.torso.Rigidbody.position-a.sequenceStartPosition;Vector2 local=Quaternion.Euler(0f,0f,-a.sequenceStartRotation)*worldDelta;a.actualPositions[reached*2]=local.x;a.actualPositions[reached*2+1]=local.y;LastActualPositions[reached]=local;a.model.RecordMeasuredProgress(worldDelta);SequenceStep=reached+1;
            int next=reached+1;if(next<5)ApplySequenceAction(a,next);else a.sequenceActive=false;
        }
    }

    private void TrainCompletedSequences()
    {
        foreach(NativeAgent a in agents)
        {
            if(a.observation==null||a.sequence==null||a.sequenceActive||a.brain==null||a.brain.IsDead)continue;
            var sample=new NativeCompletedSequence{observation=a.observation,actions=FlattenActions(a.sequence.actions),actualPositions=(float[])a.actualPositions.Clone()};
            LastM3Loss=a.model.TrainSharedM3(sample,sharedWeights,m3Optimizer);LastM3Mse=a.model.LastM3Mse;LastM3RelativeLoss=a.model.LastM3RelativeLoss;SharedM3Updates++;validation.Add(sample);if(validation.Count>ValidationCapacity)validation.RemoveAt(0);a.observation=null;a.sequence=null;
        }
    }

    private void PlanReadyAgents()
    {
        long started=System.Diagnostics.Stopwatch.GetTimestamp();M2TrainingUpdates=0;
        foreach(NativeAgent a in agents)
        {
            if(a.brain==null||a.brain.IsDead||a.identity?.torso==null||!a.ReadyForLearning||a.sequenceActive)continue;
            try
            {
                if(Tick>=a.nextFiniteCheckTick&&!a.model.Weights.IsFinite()){a.model.RepairFrom(sharedWeights);ControlFailures++;LastControlError=a.identity.creatureId+": repaired non-finite weights";}a.nextFiniteCheckTick=Tick+375;
                a.identity.torso.FillNativeGlobalInputs(a.globalInputs);Rigidbody2D body=a.identity.torso.Rigidbody;var proprio=new NativeLocalProprioception{worldPosition=body.position,worldRotationDegrees=body.rotation,localLinearVelocity=Quaternion.Euler(0f,0f,-body.rotation)*body.linearVelocity};
                a.observation=a.model.Observe(a.globalInputs,a.limbs,Tick,proprio,enableM1);a.sequence=a.model.OptimizeAndPlan(a.observation,sharedWeights,SharedM3Updates);a.sequenceStartTick=Tick;a.sequenceStartPosition=body.position;a.sequenceStartRotation=body.rotation;Array.Clear(a.actualPositions,0,a.actualPositions.Length);a.sequenceActive=true;ApplySequenceAction(a,0);M2TrainingUpdates++;
                LastM2Loss=a.model.LastM2Loss;LastM2GoalLoss=a.model.LastM2GoalLoss;LastM2EnergyLoss=a.model.LastM2EnergyLoss;GoalTicksRemaining=a.model.GoalTicksRemaining;BiasStrength=a.sequence.biasStrength;ExplorationSigma=a.sequence.explorationSigma;LeftProposalLoss=a.model.LeftProposalLoss;RightProposalLoss=a.model.RightProposalLoss;LeftProposalSamples=a.model.LeftLossSamples;RightProposalSamples=a.model.RightLossSamples;DirectionState=a.model.LatchedDirection<0?"left":a.model.LatchedDirection>0?"right":"neutral";for(int s=0;s<5;s++)LastPredictedPositions[s]=a.sequence.prediction.At(s);
            }catch(Exception e){HandleControlFailure(a,e);}
        }
        PlannerMilliseconds=(float)((System.Diagnostics.Stopwatch.GetTimestamp()-started)*1000d/System.Diagnostics.Stopwatch.Frequency);M2TrainingMilliseconds=PlannerMilliseconds;
    }

    private void ApplySequenceAction(NativeAgent a,int step)
    {
        float[] action=a.sequence.ActionAt(step);a.brain.ApplyNativeActions(action,a.sequence.goal,a.sequence.worldGoal,NativeCreatureModel.ActionTicks,Tick);float sum=0f;for(int i=0;i<a.sequence.limbCount;i++)sum+=Mathf.Abs(action[i]);LastActionMagnitude=a.sequence.limbCount==0?0f:sum/a.sequence.limbCount;
    }
    private static float[] FlattenActions(float[,] actions){float[] result=new float[NativeCreatureModel.PlanningHorizon*NativeCreatureModel.MaxLimbs];for(int s=0;s<NativeCreatureModel.PlanningHorizon;s++)for(int l=0;l<NativeCreatureModel.MaxLimbs;l++)result[s*NativeCreatureModel.MaxLimbs+l]=actions[s,l];return result;}

    private void UpdateSettlingState(){foreach(NativeAgent a in agents){if(a.hasCompletedInitialSettlement)continue;if(a.brain==null||a.brain.IsDead||a.identity?.torso?.Rigidbody==null){a.settledSupportTicks=0;continue;}Rigidbody2D b=a.identity.torso.Rigidbody;bool supported=a.identity.torso.TouchingGround||a.brain.allLimbs.Any(l=>l!=null&&l.TouchingEnvironment),slow=b.linearVelocity.sqrMagnitude<=.01f&&Mathf.Abs(b.angularVelocity)<=5f;a.settledSupportTicks=supported&&slow?Mathf.Min(10,a.settledSupportTicks+1):0;if(a.settledSupportTicks>=10)a.hasCompletedInitialSettlement=true;}}
    private void UpdateGoalProgressDisplays(){foreach(NativeAgent a in agents)if(a.goalVisualizer!=null&&a.identity?.torso!=null&&a.ReadyForLearning)a.goalVisualizer.ReportMovement(a.identity.torso.transform.position);}
    public static bool IsTemporaryGoalReached(Vector2 position,Vector2 goal)=>(position-goal).sqrMagnitude<=TemporaryGoalReachRadius*TemporaryGoalReachRadius;
    private void UpdateTemporaryGoalRewards(){if(enableM1)return;int block=Mathf.Max(0,Tick)/NativeCreatureModel.GoalWindowTicks;foreach(NativeAgent a in agents){if(a.temporaryGoalRewardBlock==block||a.observation==null||a.identity?.torso==null||a.brain==null||a.brain.IsDead)continue;if(!IsTemporaryGoalReached(a.identity.torso.Rigidbody.position,a.model.CurrentWorldGoal))continue;a.identity.torso.AddEnergy(TemporaryGoalRewardEnergy);a.temporaryGoalRewardBlock=block;Status="Temporary goal reached: +"+TemporaryGoalRewardEnergy.ToString("F0")+" energy";}}
    private void UpdateEnergyBarDisplays(){foreach(NativeAgent a in agents)if(a.energyBar!=null&&a.identity?.torso!=null)a.energyBar.Refresh(Tick);}
    private void UpdateMotionEvidence(){foreach(NativeAgent a in agents){if(a.brain==null||a.brain.IsDead||a.identity?.torso?.Rigidbody==null){a.movingEvidenceTicks=a.stillEvidenceTicks=a.motionWindowTicks=0;a.isSustainedMoving=false;continue;}if(a.motionWindowTicks<=0)a.motionWindowStartPosition=a.identity.torso.Rigidbody.position;a.motionWindowTicks++;if(a.motionWindowTicks<10)continue;float speed=(a.identity.torso.Rigidbody.position-a.motionWindowStartPosition).magnitude/Mathf.Max(.0001f,a.motionWindowTicks*Time.fixedDeltaTime);a.motionWindowStartPosition=a.identity.torso.Rigidbody.position;a.motionWindowTicks=0;if(a.isSustainedMoving){a.stillEvidenceTicks=speed<=.025f?a.stillEvidenceTicks+1:0;if(a.stillEvidenceTicks>=2)a.isSustainedMoving=false;}else{a.movingEvidenceTicks=speed>=.08f?a.movingEvidenceTicks+1:0;if(a.movingEvidenceTicks>=2)a.isSustainedMoving=true;}}}
    private void LateUpdate(){MaintainPopulationInvariant();foreach(NativeAgent a in agents.ToArray()){if(a.identity?.torso==null||a.brain==null)continue;float energy=a.identity.torso.energy;a.m1EnergyFitness+=energy-a.previousEnergy;a.previousEnergy=energy;if(a.brain.IsDead&&!a.deathRecorded){RecordDeathPenalty(a);}}if(Time.time>=nextCheckpoint){nextCheckpoint+=checkpointIntervalSeconds;SaveCheckpoint();}}
    private void HandleControlFailure(NativeAgent a,Exception e){ControlFailures++;LastControlError=a.identity==null?e.Message:a.identity.creatureId+": "+e.Message;Debug.LogError("Native control failed for creature "+LastControlError+"\n"+e);if(a.brain!=null&&!a.brain.IsDead)a.brain.Kill("native_control_error");}
    private void SeedPopulation(){for(int i=agents.Count;i<Mathf.Clamp(populationTarget,1,10);i++)Spawn(CreateSeedBody(i),"body_species_"+(i%3),"m1_seed_"+(i%3),sharedWeights.Clone(),i);}
    private void MaintainPopulationInvariant(){populationTarget=Mathf.Max(6,Mathf.Clamp(populationTarget,1,10));if(agents.Count<populationTarget)SeedPopulation();}
    private void ReplaceDeadCreaturesImmediately(){bool changed=false;foreach(NativeAgent victim in agents.Where(a=>a.brain==null||a.brain.IsDead).ToArray()){if(!agents.Contains(victim))continue;RecordDeathPenalty(victim);try{NativeAgent parent=SelectLivingParent();if(parent==null)RespawnSameGenotype(victim,victim.reservedSlot);else ReplaceAgent(victim,parent,victim.reservedSlot);replacementRetries.Remove(victim);ReplacementRetryState="none";ImmediateReplacementCount++;LastReplacementTick=Tick;changed=true;}catch(Exception e){int retries=replacementRetries.TryGetValue(victim,out int prior)?prior:0;if(retries==0){replacementRetries[victim]=1;ReplacementRetryState="queued retry for slot "+victim.reservedSlot;Debug.LogError("Native replacement failed once; retry queued: "+e.Message);}else{replacementRetries.Remove(victim);ReplacementRetryState="retry failed for slot "+victim.reservedSlot;}}}if(changed)RebuildSpeciation();}
    private NativeAgent SelectLivingParent(){Dictionary<string,float> fitness=NativeNestedSpeciation.SharedFitness(species);NativeAgent[] eligible=agents.Where(a=>a.brain!=null&&!a.brain.IsDead&&NativeCreatureModel.IsFinite(a.model.BodyFitness)&&!a.protectedOffspring).ToArray();if(eligible.Length==0)return null;NativeAgent best=null;float quality=float.NegativeInfinity;for(int i=0;i<3;i++){NativeAgent c=eligible[random.Next(eligible.Length)];float q=fitness.TryGetValue(c.identity.creatureId,out float value)?value:0f;if(q>quality){best=c;quality=q;}}return best;}
    private void RespawnSameGenotype(NativeAgent victim,int slot){BodyGenomeDto body=JsonConvert.DeserializeObject<BodyGenomeDto>(JsonConvert.SerializeObject(victim.body));NativeBrainWeights weights=victim.model.Weights.Clone();Spawn(body,victim.identity.speciesId,victim.brain.M1SpeciesId,weights,slot,victim.bodyAgeGenerations);RemoveAgentObject(victim);}
    private void ReplaceAgent(NativeAgent victim,NativeAgent parent,int slot)
    {
        BodyGenomeDto child=JsonConvert.DeserializeObject<BodyGenomeDto>(JsonConvert.SerializeObject(parent.body));int age=parent.bodyAgeGenerations+1;bool changed=bodyMutationCooldownGenerations<=0||age>=bodyMutationCooldownGenerations;if(changed){MutateBody(child);age=0;}NativeAgent m1Parent=agents.Where(a=>a.brain!=null&&!a.brain.IsDead&&a.identity.speciesId==parent.identity.speciesId).OrderByDescending(a=>a.m1EnergyFitness).FirstOrDefault()??parent;NativeBrainWeights weights=parent.model.Weights.Clone();Array.Copy(m1Parent.model.Weights.m1Input,weights.m1Input,weights.m1Input.Length);Array.Copy(m1Parent.model.Weights.m1Residual,weights.m1Residual,weights.m1Residual.Length);Array.Copy(m1Parent.model.Weights.m1Output,weights.m1Output,weights.m1Output.Length);Array.Copy(m1Parent.model.Weights.m1Bias,weights.m1Bias,weights.m1Bias.Length);weights.MutateM2(random,parent.model.IsBidirectionallyVerified?.05f:.08f);if(enableM1)weights.MutateM1(random,.04f);weights.CopyM3From(sharedWeights);Spawn(child,parent.identity.speciesId,parent.brain.M1SpeciesId,weights,slot,age);agents[agents.Count-1].protectedOffspring=changed;RemoveAgentObject(victim);Generation++;
    }
    private void RemoveAgentObject(NativeAgent victim){GameObject root=victim.identity!=null&&victim.identity.transform.parent!=null?victim.identity.transform.parent.gameObject:victim.identity?.gameObject;if(root!=null){root.SetActive(false);Destroy(root);}agents.Remove(victim);}
    private void RecordDeathPenalty(NativeAgent a){if(a==null||a.brain==null||a.deathRecorded)return;float penalty=a.brain.DeathReason=="predator"&&PredatorKillsEnabled?100f:25f;a.deathPenalty+=penalty;a.m1EnergyFitness-=penalty;a.deathRecorded=true;a.sequenceActive=false;a.observation=null;a.sequence=null;}
    private void RecoverFromExtinction(){foreach(NativeAgent a in agents){GameObject root=a.identity!=null&&a.identity.transform.parent!=null?a.identity.transform.parent.gameObject:a.identity?.gameObject;if(root!=null){root.SetActive(false);Destroy(root);}}agents.Clear();SeedPopulation();RebuildSpeciation();Status="Native ecosystem recovered from extinction";}
    private void Evolve(){if(agents.Count<2)return;NativeAgent parent=SelectLivingParent();if(parent==null)return;NativeAgent victim=agents.Where(a=>a!=parent&&!a.protectedOffspring).OrderByDescending(a=>a.model.BodyFitness).FirstOrDefault();if(victim==null)return;ReplaceAgent(victim,parent,victim.reservedSlot);MaintainPopulationInvariant();RebuildSpeciation();SaveCheckpoint();}

    private static BodyGenomeDto CreateSeedBody(int index)
    {
        var body=new BodyGenomeDto{genome_id=Guid.NewGuid().ToString("N"),limbs=new List<LimbGeneDto>()};int species=Mathf.Abs(index)%3;
        if(species==0){body.limbs.Add(NewLimb(1,0,1));body.limbs.Add(NewLimb(2,0,5));body.limbs.Add(NewLimb(3,0,7));body.limbs.Add(NewLimb(4,2,3));body.limbs.Add(NewLimb(5,3,3));body.limbs.Add(NewLimb(6,0,0));}
        else if(species==1){int[] slots={1,5,7};for(int i=0;i<3;i++)body.limbs.Add(NewLimb(i+1,0,slots[i]));for(int i=0;i<3;i++)body.limbs.Add(NewLimb(i+4,i+1,3));}
        else{body.limbs.Add(NewLimb(1,0,5));body.limbs.Add(NewLimb(2,0,7));body.limbs.Add(NewLimb(3,1,3));body.limbs.Add(NewLimb(4,2,3));body.limbs.Add(NewLimb(5,3,3));body.limbs.Add(NewLimb(6,4,3));}body.Validate();return body;
    }
    private static LimbGeneDto NewLimb(int id,int parent,int slot)=>new LimbGeneDto{innovation_id=id,parent_innovation_id=parent,attachment_slot=slot};
    private void Spawn(BodyGenomeDto body,string speciesId,string m1Species,NativeBrainWeights weights,int slot,int bodyAge=0)
    {
        body.Validate();Vector2 position=NativeSpawnPosition(slot);GameObject root=new GameObject("NativeCreature_"+body.genome_id.Substring(0,8));root.transform.position=position;CreatureIdentity identity=root.AddComponent<CreatureIdentity>();identity.creatureId=Guid.NewGuid().ToString("N");identity.speciesId=speciesId;GameObject torsoObject=new GameObject("Torso");torsoObject.transform.SetParent(root.transform);torsoObject.transform.position=position;torsoObject.AddComponent<Rigidbody2D>();torsoObject.AddComponent<SpriteRenderer>();torsoObject.AddComponent<BoxCollider2D>();Torso torso=torsoObject.AddComponent<Torso>();torso.InitFromGenome(identity,body);identity.torso=torso;identity.bodyGenomeJson=JsonConvert.SerializeObject(body);PlaceConstructedBodyOnGround(root);int layer=CreatureLayerBase+Mathf.Abs(slot)%CreatureLayerCount;SetLayerRecursively(root,layer);for(int other=CreatureLayerBase;other<CreatureLayerBase+CreatureLayerCount;other++)if(other!=layer)Physics2D.IgnoreLayerCollision(layer,other,true);Limb[] limbs=torso.GetAllLimbs().Where(l=>l!=null).Take(NativeCreatureModel.MaxLimbs).ToArray();CreatureBrain brain=torsoObject.AddComponent<CreatureBrain>();brain.Init(torso,limbs.ToList());brain.SetNativeMetadata(speciesId,m1Species);NativeGoalVisualizer visual=torsoObject.AddComponent<NativeGoalVisualizer>();visual.Initialize(torso,brain,Color.HSVToRGB(Mathf.Repeat(slot*.173f,1f),.8f,1f));NativeEnergyBar energyBar=torsoObject.AddComponent<NativeEnergyBar>();energyBar.Initialize(torso);var agent=new NativeAgent{identity=identity,brain=brain,limbs=limbs,goalVisualizer=visual,energyBar=energyBar,body=body,model=new NativeCreatureModel(weights,2003+slot),motionWindowStartPosition=torso.Rigidbody.position,previousEnergy=torso.energy,bodyAgeGenerations=Mathf.Max(0,bodyAge),reservedSlot=Mathf.Abs(slot)%NativeSpawnSlots};agents.Add(agent);
    }
    private static void PlaceConstructedBodyOnGround(GameObject root){Physics2D.SyncTransforms();Collider2D[] colliders=root.GetComponentsInChildren<Collider2D>();if(colliders.Length==0)return;float min=colliders.Min(c=>c.bounds.min.y);root.transform.position+=Vector3.up*(WorldLayout.GroundTop+.03f-min);Physics2D.SyncTransforms();}
    private static Vector2 NativeSpawnPosition(int slot)=>new Vector2(-160f+Mathf.Abs(slot)%NativeSpawnSlots*MinimumSpawnSeparation,-2f);
    private static void SetLayerRecursively(GameObject root,int layer){root.layer=layer;foreach(Transform child in root.transform)SetLayerRecursively(child.gameObject,layer);}
    private void MutateBody(BodyGenomeDto body){body.genome_id=Guid.NewGuid().ToString("N");List<LimbGeneDto> enabled=body.limbs.Where(l=>l.enabled).ToList();if(enabled.Count<NativeCreatureModel.MaxLimbs&&random.NextDouble()<.65){var sites=new List<Vector2Int>();AddFreeSites(body,0,BodyMorphologyRules.TorsoSlots,sites);foreach(LimbGeneDto parent in enabled)if(GeneDepth(body,parent.innovation_id)<NativeCreatureModel.MaxPathDepth)AddFreeSites(body,parent.innovation_id,BodyMorphologyRules.LimbSlots,sites);if(sites.Count>0){Vector2Int site=sites[random.Next(sites.Count)];body.limbs.Add(NewLimb(body.limbs.Count==0?1:body.limbs.Max(l=>l.innovation_id)+1,site.x,site.y));}}else if(enabled.Count>4&&random.NextDouble()<.35){LimbGeneDto removable=enabled.Where(l=>!body.limbs.Any(o=>o.enabled&&o.parent_innovation_id==l.innovation_id)).OrderBy(_=>random.Next()).FirstOrDefault();if(removable!=null)removable.enabled=false;}body.Validate();}
    private static void AddFreeSites(BodyGenomeDto body,int parent,IEnumerable<int> slots,List<Vector2Int> output){HashSet<int> used=body.limbs.Where(l=>l.enabled&&l.parent_innovation_id==parent).Select(l=>l.attachment_slot).ToHashSet();foreach(int slot in slots)if(!used.Contains(slot))output.Add(new Vector2Int(parent,slot));}
    private static int GeneDepth(BodyGenomeDto body,int id){int depth=0;while(id!=0){LimbGeneDto gene=body.limbs.FirstOrDefault(l=>l.enabled&&l.innovation_id==id);if(gene==null)return NativeCreatureModel.MaxPathDepth;id=gene.parent_innovation_id;depth++;}return depth;}
    private void RebuildSpeciation(){List<NativeCreatureControllerView> views=agents.Where(a=>a.identity!=null&&a.brain!=null).Select(a=>new NativeCreatureControllerView(a.identity.creatureId,a.identity.speciesId,a.brain.M1SpeciesId,a.body,a.model.Weights,a.model.BodyFitness)).ToList();species=NativeNestedSpeciation.Rebuild(views,species);Dictionary<string,NativeAgent> byId=agents.Where(a=>a.identity!=null).ToDictionary(a=>a.identity.creatureId);foreach(NativeBodySpecies body in species)foreach(NativeBrainSubSpecies sub in body.subSpecies)foreach(NativeCreatureControllerView member in sub.members)if(byId.TryGetValue(member.id,out NativeAgent a)){a.identity.speciesId=body.id;a.brain.SetNativeMetadata(body.id,sub.id);}}

    public void SaveCheckpoint()
    {
        try{if(random==null)random=new System.Random(1337^Generation^controlStep);var checkpoint=new NativeEcosystemCheckpoint{tick=Tick,generation=Generation,controlStep=controlStep,specialistSelectionCursor=specialistSelectionCursor,sharedM3Updates=SharedM3Updates,experimentId=ExperimentId,randomState=random.Next(),nextEvolutionTicks=(long)(nextEvolution*1000f),sharedWeights=sharedWeights,m3Optimizer=m3Optimizer,validation=new List<NativeCompletedSequence>(validation)};foreach(NativeAgent a in agents)checkpoint.creatures.Add(new NativeCreatureCheckpoint{creatureId=a.identity.creatureId,speciesId=a.identity.speciesId,m1SpeciesId=a.brain.M1SpeciesId,bodyGenomeJson=JsonConvert.SerializeObject(a.body),brain=a.model.Capture(),bodyFitness=float.IsInfinity(a.model.BodyFitness)?float.MaxValue:a.model.BodyFitness,deathPenalty=a.deathPenalty,m1EnergyFitness=a.m1EnergyFitness,previousEnergy=a.previousEnergy,deathRecorded=a.deathRecorded,bodyAgeGenerations=a.bodyAgeGenerations,protectedOffspring=a.protectedOffspring,temporaryGoalRewardBlock=a.temporaryGoalRewardBlock});string temp=checkpointPath+".tmp",backup=checkpointPath+".bak";Directory.CreateDirectory(Path.GetDirectoryName(checkpointPath));string json=JsonConvert.SerializeObject(checkpoint,Formatting.None);using(var stream=new FileStream(temp,FileMode.Create,FileAccess.Write,FileShare.None))using(var writer=new StreamWriter(stream)){writer.Write(json);writer.Flush();stream.Flush(true);}var check=JsonConvert.DeserializeObject<NativeEcosystemCheckpoint>(File.ReadAllText(temp));if(check==null||check.schema!=13||check.configuration!=NativeEcosystemCheckpoint.ConfigurationFingerprint)throw new InvalidDataException("Temporary v13 checkpoint failed validation");if(File.Exists(checkpointPath))File.Replace(temp,checkpointPath,backup,true);else File.Move(temp,checkpointPath);CheckpointStatus="Saved";LastCheckpointError="";LastCheckpointSavedAt=DateTime.Now.ToString("HH:mm:ss");}catch(Exception e){CheckpointStatus="Save failed";LastCheckpointError=e.Message;Debug.LogError("Native checkpoint save failed: "+e);}
    }
    private void TryLoad()
    {
        if(!File.Exists(checkpointPath))return;try{var c=JsonConvert.DeserializeObject<NativeEcosystemCheckpoint>(File.ReadAllText(checkpointPath));if(c==null||c.schema!=13||!NativeEcosystemCheckpoint.IsCompatibleConfiguration(c.configuration))throw new InvalidDataException("Native checkpoint schema/configuration mismatch");sharedWeights=c.sharedWeights??sharedWeights;m3Optimizer=c.m3Optimizer??new NativeOptimizerState();SharedM3Updates=c.sharedM3Updates;ExperimentId=string.IsNullOrWhiteSpace(c.experimentId)?ExperimentId:c.experimentId;Tick=c.tick;Generation=c.generation;controlStep=c.controlStep;specialistSelectionCursor=c.specialistSelectionCursor;random=new System.Random(unchecked((int)c.randomState));validation.Clear();if(c.validation!=null)validation.AddRange(c.validation.TakeLast(ValidationCapacity));foreach(NativeCreatureCheckpoint saved in c.creatures.Take(Mathf.Clamp(populationTarget,1,10))){BodyGenomeDto body=JsonConvert.DeserializeObject<BodyGenomeDto>(saved.bodyGenomeJson);Spawn(body,saved.speciesId,saved.m1SpeciesId,saved.brain?.weights??sharedWeights.Clone(),agents.Count,saved.bodyAgeGenerations);NativeAgent a=agents[agents.Count-1];a.model.Restore(saved.brain);a.identity.creatureId=saved.creatureId;a.deathPenalty=saved.deathPenalty;a.m1EnergyFitness=saved.m1EnergyFitness;a.previousEnergy=saved.previousEnergy;a.deathRecorded=saved.deathRecorded;a.protectedOffspring=saved.protectedOffspring;a.temporaryGoalRewardBlock=saved.temporaryGoalRewardBlock;}CheckpointStatus="Loaded";LastCheckpointSavedAt="Loaded";}catch(Exception e){LastCheckpointError=e.Message;CheckpointStatus="Rejected; fresh population";foreach(string suffix in new[]{"",".bak",".tmp"})try{string file=checkpointPath+suffix;if(File.Exists(file))File.Move(file,file+".invalid."+DateTime.UtcNow.Ticks);}catch(Exception move){LastCheckpointError+=" | "+move.Message;}}}
    private void OnApplicationQuit()=>SaveCheckpoint(); private void OnDestroy(){if(Application.isPlaying)SaveCheckpoint();}
}

public enum NativeGoalMovement{Still,Toward,Away}
public sealed class NativeGoalVisualizer:MonoBehaviour
{
    public const float MovementThreshold=.05f,MaximumDisplayDistance=4f;private Torso torso;private CreatureBrain brain;private GameObject root;private Transform marker;private LineRenderer line;private TextMesh label,movement;private Material material;private Color color;private Vector2 goal,displayGoal,baseline;private bool hasGoal,hasBaseline;
    public void Initialize(Torso t,CreatureBrain b,Color c){torso=t;brain=b;color=c;Ensure();Refresh();}
    private void LateUpdate(){if(torso==null||brain==null)return;Ensure();Refresh();if(hasGoal&&root.activeSelf){UpdateDisplayCue();line.SetPosition(0,torso.transform.position);line.SetPosition(1,displayGoal);}}
    private void Ensure(){if(marker!=null)return;root=new GameObject("NativeGoalVisual");root.SetActive(false);GameObject m=new GameObject("NativeGoalMarker");m.transform.SetParent(root.transform,false);var sr=m.AddComponent<SpriteRenderer>();sr.sprite=BodyUtils.GetSquareSprite();sr.color=new Color(color.r,color.g,color.b,.72f);sr.sortingOrder=-10;m.transform.localScale=new Vector3(.22f,.22f,1f);marker=m.transform;GameObject l=new GameObject("NativeGoalLine");l.transform.SetParent(root.transform,false);line=l.AddComponent<LineRenderer>();line.positionCount=2;line.useWorldSpace=true;line.startWidth=.025f;line.endWidth=.012f;line.sortingOrder=-11;material=new Material(Shader.Find("Sprites/Default"));material.color=color;line.sharedMaterial=material;label=MakeText("NativeGoalCoordinates",-9);movement=MakeText("NativeGoalMovement",-8);movement.text="MEASURING";}
    private TextMesh MakeText(string n,int order){GameObject o=new GameObject(n);o.transform.SetParent(root.transform,false);TextMesh t=o.AddComponent<TextMesh>();t.anchor=TextAnchor.MiddleCenter;t.alignment=TextAlignment.Center;t.fontSize=48;t.characterSize=.08f;t.fontStyle=FontStyle.Bold;t.color=Color.white;t.GetComponent<MeshRenderer>().sortingOrder=order;return t;}
    private void Refresh(){if(!brain.HasWorldGoal){if(root.activeSelf)root.SetActive(false);return;}if(hasGoal&&(brain.CurrentWorldGoal-goal).sqrMagnitude<=1e-6f)return;goal=brain.CurrentWorldGoal;hasGoal=true;hasBaseline=false;root.SetActive(true);UpdateDisplayCue();}
    private void UpdateDisplayCue(){Vector2 origin=torso.transform.position;displayGoal=DisplayTarget(origin,goal);marker.position=goal;label.text=$"G ({goal.x:F2}, {goal.y:F2})";label.transform.position=goal+Vector2.up*.4f;movement.transform.position=goal+Vector2.up;}
    public void ReportMovement(Vector2 current){Ensure();Refresh();if(!brain.HasWorldGoal)return;if(!hasBaseline){baseline=current;hasBaseline=true;movement.text="MEASURING";movement.color=Color.white;return;}Vector2 toward=goal-baseline;float p=toward.sqrMagnitude<=1e-6f?0f:Vector2.Dot(current-baseline,toward.normalized);baseline=current;NativeGoalMovement s=ClassifyProgress(p);movement.text=s==NativeGoalMovement.Toward?$"TOWARD +{p:F2}":s==NativeGoalMovement.Away?$"AWAY {p:F2}":$"STILL {p:+0.00;-0.00;0.00}";movement.color=MovementColor(s);}
    public static NativeGoalMovement ClassifyProgress(float p)=>p>MovementThreshold?NativeGoalMovement.Toward:p<-MovementThreshold?NativeGoalMovement.Away:NativeGoalMovement.Still;
    public static Color MovementColor(NativeGoalMovement movement)=>movement==NativeGoalMovement.Toward?new Color(.2f,1f,.3f):movement==NativeGoalMovement.Away?new Color(1f,.22f,.22f):new Color(1f,.85f,.2f);
    public static Vector2 DisplayTarget(Vector2 origin,Vector2 actualGoal){Vector2 offset=actualGoal-origin;return offset.sqrMagnitude<=MaximumDisplayDistance*MaximumDisplayDistance?actualGoal:origin+offset.normalized*MaximumDisplayDistance;}
    private void OnDestroy(){if(material!=null){if(Application.isPlaying)Destroy(material);else DestroyImmediate(material);}if(root!=null){if(Application.isPlaying)Destroy(root);else DestroyImmediate(root);}}
}

public sealed class NativeEnergyBar:MonoBehaviour
{
    public const int RefreshIntervalTicks=50;private Torso torso;private GameObject root;private Transform fill;private SpriteRenderer fillRenderer;private float width;public int LastRefreshTick{get;private set;}=-1;
    public void Initialize(Torso value){torso=value;Ensure();Refresh();}
    public void Refresh(int tick){Refresh();LastRefreshTick=tick;}
    public void Refresh(){if(torso==null)return;Ensure();float fraction=EnergyFraction(torso.energy,torso.maxEnergy);float height=.12f;root.transform.position=(Vector2)torso.transform.position+Vector2.up*(Mathf.Max(.25f,torso.dimensions.y*.5f)+.38f);fill.localScale=new Vector3(width*fraction,height,1f);fill.localPosition=new Vector3(-width*.5f+width*fraction*.5f,0f,-.01f);fillRenderer.color=EnergyColor(fraction);}
    public static float EnergyFraction(float energy,float maximum)=>maximum<=0f?0f:Mathf.Clamp01(energy/maximum);
    public static Color EnergyColor(float fraction)=>Color.Lerp(new Color(1f,.2f,.2f),new Color(.2f,1f,.3f),Mathf.Clamp01(fraction));
    private void Ensure(){if(root!=null)return;width=Mathf.Max(.8f,torso.dimensions.x*1.8f);root=new GameObject("NativeEnergyBarVisual");GameObject foreground=new GameObject("Fill");foreground.transform.SetParent(root.transform,false);fill=foreground.transform;fillRenderer=foreground.AddComponent<SpriteRenderer>();fillRenderer.sprite=BodyUtils.GetSquareSprite();fillRenderer.sortingOrder=21;}
    private void OnDestroy(){if(root!=null){if(Application.isPlaying)Destroy(root);else DestroyImmediate(root);}}
}
