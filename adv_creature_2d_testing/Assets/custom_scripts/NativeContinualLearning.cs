using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using Newtonsoft.Json;

[Serializable]
public sealed class NativeSynapticSlot
{
    public string name;
    [JsonIgnore] public float[] importance,reference,path,start,gradient;

    public NativeSynapticSlot Clone()=>new NativeSynapticSlot{name=name,importance=Copy(importance),reference=Copy(reference),path=Copy(path),start=Copy(start),gradient=Copy(gradient)};
    private static float[] Copy(float[] value)=>value==null?null:(float[])value.Clone();
}

[Serializable]
public sealed class NativeSynapticIntelligence
{
    public const int WarmupUpdates=256,ConsolidationInterval=128;
    public const float Strength=.2f,ImportanceCap=10f,Epsilon=1e-3f;
    [JsonIgnore] public List<NativeSynapticSlot> slots=new List<NativeSynapticSlot>();

    public NativeSynapticIntelligence Clone(){var copy=new NativeSynapticIntelligence();foreach(var slot in slots??new List<NativeSynapticSlot>())copy.slots.Add(slot.Clone());return copy;}
    public NativeSynapticSlot Get(string name,int length,float[] parameter)
    {
        if(slots==null)slots=new List<NativeSynapticSlot>();NativeSynapticSlot slot=slots.FirstOrDefault(s=>s.name==name);
        if(slot==null){slot=new NativeSynapticSlot{name=name};slots.Add(slot);}if(slot.importance==null||slot.importance.Length!=length){slot.importance=new float[length];slot.reference=(float[])parameter.Clone();slot.path=new float[length];slot.start=(float[])parameter.Clone();slot.gradient=new float[length];}return slot;
    }
    public void AddPenalty(IReadOnlyList<KeyValuePair<string,float[]>> parameters,Dictionary<string,float[]> gradients)
    {foreach(var parameter in parameters){NativeSynapticSlot slot=Get(parameter.Key,parameter.Value.Length,parameter.Value);float[] g=gradients[parameter.Key];for(int i=0;i<g.Length;i++)g[i]+=2f*Strength*slot.importance[i]*(parameter.Value[i]-slot.reference[i]);}}
    public void BeginStep(IReadOnlyList<KeyValuePair<string,float[]>> parameters,Dictionary<string,float[]> gradients)
    {foreach(var parameter in parameters){NativeSynapticSlot slot=Get(parameter.Key,parameter.Value.Length,parameter.Value);Array.Copy(parameter.Value,slot.start,parameter.Value.Length);Array.Copy(gradients[parameter.Key],slot.gradient,parameter.Value.Length);}}
    public void EndStep(IReadOnlyList<KeyValuePair<string,float[]>> parameters,int optimizerStep)
    {
        foreach(var parameter in parameters){NativeSynapticSlot slot=Get(parameter.Key,parameter.Value.Length,parameter.Value);for(int i=0;i<parameter.Value.Length;i++)slot.path[i]+=-slot.gradient[i]*(parameter.Value[i]-slot.start[i]);}
        if(optimizerStep<WarmupUpdates||optimizerStep%ConsolidationInterval!=0)return;
        foreach(var parameter in parameters){NativeSynapticSlot slot=Get(parameter.Key,parameter.Value.Length,parameter.Value);float[] contribution=new float[parameter.Value.Length];float maximum=0f;for(int i=0;i<parameter.Value.Length;i++){float delta=parameter.Value[i]-slot.reference[i];contribution[i]=Math.Max(0f,slot.path[i])/(delta*delta+Epsilon);maximum=Math.Max(maximum,contribution[i]);}float scale=maximum>ImportanceCap?ImportanceCap/maximum:1f;for(int i=0;i<parameter.Value.Length;i++)slot.importance[i]=Math.Min(ImportanceCap,slot.importance[i]+contribution[i]*scale);Array.Copy(parameter.Value,slot.reference,parameter.Value.Length);Array.Clear(slot.path,0,slot.path.Length);}
    }
    public bool IsFinite()=>slots==null||slots.All(s=>Finite(s.importance)&&Finite(s.reference)&&Finite(s.path));
    private static bool Finite(float[] values)=>values!=null&&values.All(NativeCreatureModel.IsFinite);
}

[Serializable]
public sealed class NativeReplayBuffer
{
    public const int Capacity=16384;
    public long seen;
    public NativeDeterministicRng random=new NativeDeterministicRng(0xC01DF00DUL);
    [JsonIgnore] public List<NativeCompletedSequence> samples=new List<NativeCompletedSequence>();
    [JsonIgnore] private Dictionary<string,List<int>> compatible=new Dictionary<string,List<int>>(StringComparer.Ordinal);
    public int Count=>samples?.Count??0;

    public void Offer(NativeCompletedSequence sample)
    {
        if(sample?.observation==null)return;seen++;NativeCompletedSequence copy=sample.Clone();if(samples.Count<Capacity){samples.Add(copy);Index(samples.Count-1,copy.bodySignature);return;}long candidate=NextLongBelow(seen);if(candidate<Capacity){int slot=(int)candidate;Unindex(slot,samples[slot]?.bodySignature);samples[slot]=copy;Index(slot,copy.bodySignature);}
    }
    public List<NativeCompletedSequence> Sample(int count,string bodySignature=null)
    {
        EnsureIndex();List<int> eligible=bodySignature==null?null:(compatible.TryGetValue(bodySignature,out var indexed)?indexed:null);int available=eligible==null?(bodySignature==null?samples.Count:0):eligible.Count;var result=new List<NativeCompletedSequence>(Math.Min(count,available));var selected=new HashSet<int>();while(result.Count<count&&selected.Count<available){int slot=eligible==null?random.NextInt(samples.Count):eligible[random.NextInt(eligible.Count)];if(selected.Add(slot))result.Add(samples[slot]);}return result;
    }
    public NativeReplayBuffer Clone(){var copy=new NativeReplayBuffer{seen=seen,random=new NativeDeterministicRng(random.State)};foreach(var sample in samples??new List<NativeCompletedSequence>()){copy.samples.Add(sample.Clone());copy.Index(copy.samples.Count-1,sample.bodySignature);}return copy;}
    public void RebuildIndex(){compatible=new Dictionary<string,List<int>>(StringComparer.Ordinal);for(int i=0;i<samples.Count;i++)Index(i,samples[i]?.bodySignature);}
    private void EnsureIndex(){if(compatible==null||compatible.Count==0&&samples.Count>0)RebuildIndex();}
    private void Index(int slot,string signature){signature=signature??string.Empty;if(!compatible.TryGetValue(signature,out var list))compatible[signature]=list=new List<int>();list.Add(slot);}
    private void Unindex(int slot,string signature){signature=signature??string.Empty;if(!compatible.TryGetValue(signature,out var list))return;list.Remove(slot);if(list.Count==0)compatible.Remove(signature);}
    private long NextLongBelow(long exclusiveMax)=>random.NextLong(exclusiveMax);
    public static NativeObservation CloneObservation(NativeObservation value)
    {
        if(value==null)return null;return new NativeObservation{torsoWithGoal=Copy(value.torsoWithGoal),torsoWithoutGoal=Copy(value.torsoWithoutGoal),jointFeatures=Copy(value.jointFeatures),positionEncodings=Copy(value.positionEncodings),limbCount=value.limbCount,tick=value.tick,episode=value.episode,controllerId=value.controllerId,bodySignature=value.bodySignature,practice=value.practice,goalX=value.goalX,goalY=value.goalY,goalHeadingSin=value.goalHeadingSin,goalHeadingCos=value.goalHeadingCos,worldGoalX=value.worldGoalX,worldGoalY=value.worldGoalY,worldGoalHeadingSin=value.worldGoalHeadingSin,worldGoalHeadingCos=value.worldGoalHeadingCos,startWorldX=value.startWorldX,startWorldY=value.startWorldY,startWorldRotationDegrees=value.startWorldRotationDegrees};
    }
    private static float[] Copy(float[] values)=>values==null?null:(float[])values.Clone();
}

public sealed class NativeTrainingResult
{
    public string controllerId,error;public NativeM2BatchMetrics m2;public NativeM3BatchMetrics m3;public long milliseconds;
}

public sealed class NativeContinualLearner:IDisposable
{
    private const int MaximumQueuedSequences=64; public const int M3WarmupFreshSequences=256;
    private readonly ConcurrentQueue<NativeCompletedSequence> queue=new ConcurrentQueue<NativeCompletedSequence>();
    private readonly ConcurrentQueue<NativeTrainingResult> results=new ConcurrentQueue<NativeTrainingResult>();
    private readonly AutoResetEvent signal=new AutoResetEvent(false);private readonly Thread worker;private volatile bool stopping;private int queued;
    private readonly object stateLock=new object();private NativeDynamicsWeights sharedM3;private NativeOptimizerState m3Optimizer;private NativeSynapticIntelligence m3Protection;private NativeReplayBuffer replay;
    private readonly Func<string,NativeCreatureModel> resolveModel;
    public int QueueDepth=>Volatile.Read(ref queued);public int Dropped{get;private set;}public bool Busy{get;private set;}public NativeDynamicsWeights SharedM3{get{lock(stateLock)return sharedM3;}}
    public int ReplayCount{get{lock(stateLock)return replay.Count;}}

    public NativeContinualLearner(NativeDynamicsWeights dynamics,NativeOptimizerState optimizer,NativeSynapticIntelligence protection,NativeReplayBuffer buffer,Func<string,NativeCreatureModel> resolver)
    {sharedM3=dynamics??NativeDynamicsWeights.Create(7331);m3Optimizer=optimizer??new NativeOptimizerState();m3Protection=protection??new NativeSynapticIntelligence();replay=buffer??new NativeReplayBuffer();resolveModel=resolver;worker=new Thread(Run){IsBackground=true,Name="Native continual learner"};worker.Start();}
    public bool Enqueue(NativeCompletedSequence sample){if(sample==null)return false;if(Interlocked.Increment(ref queued)>MaximumQueuedSequences){Interlocked.Decrement(ref queued);Dropped++;return false;}queue.Enqueue(sample.Clone());signal.Set();return true;}
    public bool TryResult(out NativeTrainingResult result)=>results.TryDequeue(out result);
    private void Run()
    {
        while(!stopping){if(!queue.TryDequeue(out NativeCompletedSequence fresh)){signal.WaitOne(50);continue;}Interlocked.Decrement(ref queued);Busy=true;long started=System.Diagnostics.Stopwatch.GetTimestamp();var result=new NativeTrainingResult{controllerId=fresh.controllerId};try{
            NativeCreatureModel model=resolveModel?.Invoke(fresh.controllerId);NativeDynamicsWeights m3Snapshot;List<NativeCompletedSequence> m3Batch,m2Replay;bool m2Ready;lock(stateLock){m2Ready=replay.seen>=M3WarmupFreshSequences;m3Batch=new List<NativeCompletedSequence>{fresh};m3Batch.AddRange(replay.Sample(3));m2Replay=replay.Sample(3,fresh.bodySignature);replay.Offer(fresh);m3Snapshot=sharedM3.Clone();}
            NativeOptimizerState optimizer;NativeSynapticIntelligence protection;lock(stateLock){optimizer=m3Optimizer.Clone();protection=m3Protection.Clone();}result.m3=NativeCreatureModel.TrainSharedM3Batch(m3Batch,m3Snapshot,optimizer,protection);if(!m3Snapshot.IsFinite()||!protection.IsFinite())throw new InvalidOperationException("M3 update produced invalid state");
            if(m2Ready&&model!=null){var observations=new List<NativeObservation>{fresh.observation};observations.AddRange(m2Replay.Select(s=>s.observation));result.m2=model.TrainM2Batch(observations,m3Snapshot,fresh.observation.tick);}
            lock(stateLock){sharedM3=m3Snapshot;m3Optimizer=optimizer;m3Protection=protection;}
        }catch(Exception e){result.error=e.ToString();}finally{result.milliseconds=(long)((System.Diagnostics.Stopwatch.GetTimestamp()-started)*1000d/System.Diagnostics.Stopwatch.Frequency);results.Enqueue(result);Busy=false;}}
    }
    public void Capture(out NativeDynamicsWeights dynamics,out NativeOptimizerState optimizer,out NativeSynapticIntelligence protection,out NativeReplayBuffer buffer){lock(stateLock){dynamics=sharedM3.Clone();optimizer=m3Optimizer.Clone();protection=m3Protection.Clone();buffer=replay.Clone();}}
    public void Dispose(){stopping=true;signal.Set();if(worker.IsAlive)worker.Join(2000);signal.Dispose();}
}
