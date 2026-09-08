using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;

public static class NativeTensorCheckpointStore
{
    private const int Magic=0x4E543230;

    public static void Write(string path,NativeEcosystemCheckpoint checkpoint)
    {
        using(var stream=new FileStream(path,FileMode.Create,FileAccess.Write,FileShare.None))
        using(var writer=new BinaryWriter(stream,Encoding.UTF8,true))
        {
            writer.Write(Magic);writer.Write(20);writer.Write(NativeEcosystemCheckpoint.ConfigurationFingerprint);
            WriteParameters(writer,checkpoint.sharedM3.Parameters());WriteOptimizer(writer,checkpoint.m3Optimizer,checkpoint.sharedM3.Parameters());WriteProtection(writer,checkpoint.m3Protection,checkpoint.sharedM3.Parameters());
            writer.Write(checkpoint.creatures.Count);
            foreach(NativeCreatureCheckpoint creature in checkpoint.creatures){writer.Write(creature.creatureId??string.Empty);WriteParameters(writer,PolicyParameters(creature.brain.weights));WriteOptimizer(writer,creature.brain.m2Optimizer,creature.brain.weights.M2Parameters());WriteProtection(writer,creature.brain.m2Protection,creature.brain.weights.M2Parameters());}
            WriteReplay(writer,checkpoint.replay);
            writer.Flush();stream.Flush(true);
        }
    }

    public static void Read(string path,NativeEcosystemCheckpoint checkpoint)
    {
        using(var stream=new FileStream(path,FileMode.Open,FileAccess.Read,FileShare.Read))
        using(var reader=new BinaryReader(stream,Encoding.UTF8,true))
        {
            if(reader.ReadInt32()!=Magic||reader.ReadInt32()!=20||reader.ReadString()!=NativeEcosystemCheckpoint.ConfigurationFingerprint)throw new InvalidDataException("v20 tensor header mismatch");
            checkpoint.sharedM3=NativeDynamicsWeights.Create(7331);ReadParameters(reader,checkpoint.sharedM3.Parameters());checkpoint.m3Optimizer=checkpoint.m3Optimizer??new NativeOptimizerState();ReadOptimizer(reader,checkpoint.m3Optimizer,checkpoint.sharedM3.Parameters());checkpoint.m3Protection=checkpoint.m3Protection??new NativeSynapticIntelligence();ReadProtection(reader,checkpoint.m3Protection,checkpoint.sharedM3.Parameters());
            int count=reader.ReadInt32();if(count!=checkpoint.creatures.Count)throw new InvalidDataException("v20 tensor creature count mismatch");
            var byId=checkpoint.creatures.ToDictionary(c=>c.creatureId??string.Empty,StringComparer.Ordinal);
            for(int i=0;i<count;i++){string id=reader.ReadString();if(!byId.TryGetValue(id,out NativeCreatureCheckpoint creature))throw new InvalidDataException("v20 tensor creature identity mismatch");creature.brain=creature.brain??new NativeCreatureSave();creature.brain.weights=NativeBrainWeights.Create(1800+i);ReadParameters(reader,PolicyParameters(creature.brain.weights));creature.brain.m2Optimizer=creature.brain.m2Optimizer??new NativeOptimizerState();ReadOptimizer(reader,creature.brain.m2Optimizer,creature.brain.weights.M2Parameters());creature.brain.m2Protection=creature.brain.m2Protection??new NativeSynapticIntelligence();ReadProtection(reader,creature.brain.m2Protection,creature.brain.weights.M2Parameters());}
            checkpoint.replay=ReadReplay(reader);if(checkpoint.replay.Count!=checkpoint.replayCount)throw new InvalidDataException("v20 replay count mismatch");if(stream.Position!=stream.Length)throw new InvalidDataException("v20 tensor file has trailing bytes");
        }
    }

    public static string Sha256(string path){using(var sha=SHA256.Create())using(var stream=File.OpenRead(path)){byte[] hash=sha.ComputeHash(stream);return string.Concat(hash.Select(b=>b.ToString("x2")));}}
    public static void ValidateFileName(string fileName){if(string.IsNullOrWhiteSpace(fileName)||Path.GetFileName(fileName)!=fileName||!fileName.StartsWith("native_ecosystem_v20.",StringComparison.Ordinal)||!fileName.EndsWith(".bin",StringComparison.Ordinal))throw new InvalidDataException("Unsafe v20 tensor filename");}

    private static IEnumerable<KeyValuePair<string,float[]>> PolicyParameters(NativeBrainWeights weights)
    {
        yield return P("m1Input",weights.m1Input);yield return P("m1Residual",weights.m1Residual);yield return P("m1Output",weights.m1Output);yield return P("m1Bias",weights.m1Bias);
        foreach(var parameter in weights.M2Parameters())yield return P("m2."+parameter.Key,parameter.Value);
    }
    private static KeyValuePair<string,float[]> P(string name,float[] value)=>new KeyValuePair<string,float[]>(name,value);
    private static void WriteParameters(BinaryWriter writer,IEnumerable<KeyValuePair<string,float[]>> source){var parameters=source.ToList();writer.Write(parameters.Count);foreach(var parameter in parameters){writer.Write(parameter.Key);writer.Write(parameter.Value.Length);foreach(float value in parameter.Value)writer.Write(value);}}
    private static void ReadParameters(BinaryReader reader,IEnumerable<KeyValuePair<string,float[]>> source){var expected=source.ToDictionary(p=>p.Key,p=>p.Value,StringComparer.Ordinal);int count=reader.ReadInt32();if(count!=expected.Count)throw new InvalidDataException("v20 tensor parameter count mismatch");for(int p=0;p<count;p++){string name=reader.ReadString();int length=reader.ReadInt32();if(!expected.TryGetValue(name,out float[] values)||values.Length!=length)throw new InvalidDataException("v20 tensor shape mismatch: "+name);for(int i=0;i<length;i++)values[i]=reader.ReadSingle();}}
    private static void WriteOptimizer(BinaryWriter writer,NativeOptimizerState optimizer,IEnumerable<KeyValuePair<string,float[]>> parameters){optimizer=optimizer??new NativeOptimizerState();writer.Write(optimizer.step);var list=parameters.ToList();writer.Write(list.Count);foreach(var parameter in list){NativeOptimizerSlot slot=optimizer.Get(parameter.Key,parameter.Value.Length);writer.Write(parameter.Key);writer.Write(parameter.Value.Length);foreach(float value in slot.first)writer.Write(value);foreach(float value in slot.second)writer.Write(value);}}
    private static void ReadOptimizer(BinaryReader reader,NativeOptimizerState optimizer,IEnumerable<KeyValuePair<string,float[]>> parameters){optimizer.step=reader.ReadInt32();var expected=parameters.ToDictionary(p=>p.Key,p=>p.Value.Length,StringComparer.Ordinal);int count=reader.ReadInt32();if(count!=expected.Count)throw new InvalidDataException("v20 optimizer slot count mismatch");optimizer.slots=new List<NativeOptimizerSlot>();for(int p=0;p<count;p++){string name=reader.ReadString();int length=reader.ReadInt32();if(!expected.TryGetValue(name,out int expectedLength)||length!=expectedLength)throw new InvalidDataException("v20 optimizer shape mismatch: "+name);NativeOptimizerSlot slot=optimizer.Get(name,length);for(int i=0;i<length;i++)slot.first[i]=reader.ReadSingle();for(int i=0;i<length;i++)slot.second[i]=reader.ReadSingle();}}
    private static void WriteProtection(BinaryWriter writer,NativeSynapticIntelligence protection,IEnumerable<KeyValuePair<string,float[]>> parameters){protection=protection??new NativeSynapticIntelligence();var list=parameters.ToList();writer.Write(list.Count);foreach(var parameter in list){NativeSynapticSlot slot=protection.Get(parameter.Key,parameter.Value.Length,parameter.Value);writer.Write(parameter.Key);writer.Write(parameter.Value.Length);WriteArray(writer,slot.importance);WriteArray(writer,slot.reference);WriteArray(writer,slot.path);}}
    private static void ReadProtection(BinaryReader reader,NativeSynapticIntelligence protection,IEnumerable<KeyValuePair<string,float[]>> parameters){var expected=parameters.ToDictionary(p=>p.Key,p=>p.Value,StringComparer.Ordinal);int count=reader.ReadInt32();if(count!=expected.Count)throw new InvalidDataException("v20 protection slot count mismatch");protection.slots=new List<NativeSynapticSlot>();for(int p=0;p<count;p++){string name=reader.ReadString();int length=reader.ReadInt32();if(!expected.TryGetValue(name,out float[] parameter)||parameter.Length!=length)throw new InvalidDataException("v20 protection shape mismatch: "+name);NativeSynapticSlot slot=protection.Get(name,length,parameter);ReadArray(reader,slot.importance);ReadArray(reader,slot.reference);ReadArray(reader,slot.path);}}
    private static void WriteReplay(BinaryWriter writer,List<NativeCompletedSequence> samples){samples=samples??new List<NativeCompletedSequence>();writer.Write(samples.Count);foreach(var sample in samples){writer.Write(sample.sampleId??string.Empty);writer.Write(sample.bodySignature??string.Empty);writer.Write(sample.controllerId??string.Empty);WriteObservation(writer,sample.observation);WriteArray(writer,sample.actions);WriteArray(writer,sample.actualPoses);WriteArray(writer,sample.actualAuxiliary);}}
    private static List<NativeCompletedSequence> ReadReplay(BinaryReader reader){int count=reader.ReadInt32();if(count<0||count>NativeReplayBuffer.Capacity)throw new InvalidDataException("v20 replay capacity invalid");var result=new List<NativeCompletedSequence>(count);for(int i=0;i<count;i++)result.Add(new NativeCompletedSequence{sampleId=reader.ReadString(),bodySignature=reader.ReadString(),controllerId=reader.ReadString(),observation=ReadObservation(reader),actions=ReadArray(reader,NativeCreatureModel.PlanningHorizon*NativeCreatureModel.MaxLimbs),actualPoses=ReadArray(reader,NativeCreatureModel.M3OutputSize),actualAuxiliary=ReadArray(reader,NativeCreatureModel.M3AuxiliaryOutputSize)});return result;}
    private static void WriteObservation(BinaryWriter writer,NativeObservation o){if(o==null)throw new InvalidDataException("Cannot persist null replay observation");writer.Write(o.limbCount);writer.Write(o.tick);writer.Write(o.controllerId??string.Empty);writer.Write(o.bodySignature??string.Empty);writer.Write(o.practice);foreach(float value in new[]{o.goalX,o.goalY,o.goalHeadingSin,o.goalHeadingCos,o.worldGoalX,o.worldGoalY,o.worldGoalHeadingSin,o.worldGoalHeadingCos,o.startWorldX,o.startWorldY,o.startWorldRotationDegrees})writer.Write(value);WriteArray(writer,o.torsoWithGoal);WriteArray(writer,o.torsoWithoutGoal);WriteArray(writer,o.jointFeatures);WriteArray(writer,o.positionEncodings);}
    private static NativeObservation ReadObservation(BinaryReader reader){var o=new NativeObservation{limbCount=reader.ReadInt32(),tick=reader.ReadInt32(),controllerId=reader.ReadString(),bodySignature=reader.ReadString(),practice=reader.ReadBoolean()};o.goalX=reader.ReadSingle();o.goalY=reader.ReadSingle();o.goalHeadingSin=reader.ReadSingle();o.goalHeadingCos=reader.ReadSingle();o.worldGoalX=reader.ReadSingle();o.worldGoalY=reader.ReadSingle();o.worldGoalHeadingSin=reader.ReadSingle();o.worldGoalHeadingCos=reader.ReadSingle();o.startWorldX=reader.ReadSingle();o.startWorldY=reader.ReadSingle();o.startWorldRotationDegrees=reader.ReadSingle();o.torsoWithGoal=ReadArray(reader,NativeCreatureModel.M2ContextSize);o.torsoWithoutGoal=ReadArray(reader,NativeCreatureModel.M3ContextSize);o.jointFeatures=ReadArray(reader,NativeCreatureModel.MaxLimbs*NativeCreatureModel.JointFeatureSize);o.positionEncodings=ReadArray(reader,NativeCreatureModel.MaxLimbs*NativeCreatureModel.PositionEncodingSize);return o;}
    private static void WriteArray(BinaryWriter writer,float[] values){if(values==null)throw new InvalidDataException("Cannot persist null tensor");writer.Write(values.Length);foreach(float value in values){if(!NativeCreatureModel.IsFinite(value))throw new InvalidDataException("Cannot persist non-finite tensor");writer.Write(value);}}
    private static float[] ReadArray(BinaryReader reader,int expected){int length=reader.ReadInt32();if(length!=expected)throw new InvalidDataException("v20 replay tensor shape mismatch");var values=new float[length];for(int i=0;i<length;i++)values[i]=reader.ReadSingle();return values;}
    private static void ReadArray(BinaryReader reader,float[] target){int length=reader.ReadInt32();if(target==null||length!=target.Length)throw new InvalidDataException("v20 state tensor shape mismatch");for(int i=0;i<length;i++)target[i]=reader.ReadSingle();}
}
