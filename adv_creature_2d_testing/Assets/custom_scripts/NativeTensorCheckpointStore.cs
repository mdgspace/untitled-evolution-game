using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;

public static class NativeTensorCheckpointStore
{
    private const int Magic=0x4E543138;

    public static void Write(string path,NativeEcosystemCheckpoint checkpoint)
    {
        using(var stream=new FileStream(path,FileMode.Create,FileAccess.Write,FileShare.None))
        using(var writer=new BinaryWriter(stream,Encoding.UTF8,true))
        {
            writer.Write(Magic);writer.Write(18);writer.Write(NativeEcosystemCheckpoint.ConfigurationFingerprint);
            WriteParameters(writer,checkpoint.sharedM3.Parameters());WriteOptimizer(writer,checkpoint.m3Optimizer,checkpoint.sharedM3.Parameters());
            writer.Write(checkpoint.creatures.Count);
            foreach(NativeCreatureCheckpoint creature in checkpoint.creatures){writer.Write(creature.creatureId??string.Empty);WriteParameters(writer,PolicyParameters(creature.brain.weights));WriteOptimizer(writer,creature.brain.m2Optimizer,creature.brain.weights.M2Parameters());}
            writer.Flush();stream.Flush(true);
        }
    }

    public static void Read(string path,NativeEcosystemCheckpoint checkpoint)
    {
        using(var stream=new FileStream(path,FileMode.Open,FileAccess.Read,FileShare.Read))
        using(var reader=new BinaryReader(stream,Encoding.UTF8,true))
        {
            if(reader.ReadInt32()!=Magic||reader.ReadInt32()!=18||reader.ReadString()!=NativeEcosystemCheckpoint.ConfigurationFingerprint)throw new InvalidDataException("v18 tensor header mismatch");
            checkpoint.sharedM3=NativeDynamicsWeights.Create(7331);ReadParameters(reader,checkpoint.sharedM3.Parameters());checkpoint.m3Optimizer=checkpoint.m3Optimizer??new NativeOptimizerState();ReadOptimizer(reader,checkpoint.m3Optimizer,checkpoint.sharedM3.Parameters());
            int count=reader.ReadInt32();if(count!=checkpoint.creatures.Count)throw new InvalidDataException("v18 tensor creature count mismatch");
            var byId=checkpoint.creatures.ToDictionary(c=>c.creatureId??string.Empty,StringComparer.Ordinal);
            for(int i=0;i<count;i++){string id=reader.ReadString();if(!byId.TryGetValue(id,out NativeCreatureCheckpoint creature))throw new InvalidDataException("v18 tensor creature identity mismatch");creature.brain=creature.brain??new NativeCreatureSave();creature.brain.weights=NativeBrainWeights.Create(1800+i);ReadParameters(reader,PolicyParameters(creature.brain.weights));creature.brain.m2Optimizer=creature.brain.m2Optimizer??new NativeOptimizerState();ReadOptimizer(reader,creature.brain.m2Optimizer,creature.brain.weights.M2Parameters());}
            if(stream.Position!=stream.Length)throw new InvalidDataException("v18 tensor file has trailing bytes");
        }
    }

    public static string Sha256(string path){using(var sha=SHA256.Create())using(var stream=File.OpenRead(path)){byte[] hash=sha.ComputeHash(stream);return string.Concat(hash.Select(b=>b.ToString("x2")));}}
    public static void ValidateFileName(string fileName){if(string.IsNullOrWhiteSpace(fileName)||Path.GetFileName(fileName)!=fileName||!fileName.StartsWith("native_ecosystem_v18.",StringComparison.Ordinal)||!fileName.EndsWith(".bin",StringComparison.Ordinal))throw new InvalidDataException("Unsafe v18 tensor filename");}

    private static IEnumerable<KeyValuePair<string,float[]>> PolicyParameters(NativeBrainWeights weights)
    {
        yield return P("m1Input",weights.m1Input);yield return P("m1Residual",weights.m1Residual);yield return P("m1Output",weights.m1Output);yield return P("m1Bias",weights.m1Bias);
        foreach(var parameter in weights.M2Parameters())yield return P("m2."+parameter.Key,parameter.Value);
    }
    private static KeyValuePair<string,float[]> P(string name,float[] value)=>new KeyValuePair<string,float[]>(name,value);
    private static void WriteParameters(BinaryWriter writer,IEnumerable<KeyValuePair<string,float[]>> source){var parameters=source.ToList();writer.Write(parameters.Count);foreach(var parameter in parameters){writer.Write(parameter.Key);writer.Write(parameter.Value.Length);foreach(float value in parameter.Value)writer.Write(value);}}
    private static void ReadParameters(BinaryReader reader,IEnumerable<KeyValuePair<string,float[]>> source){var expected=source.ToDictionary(p=>p.Key,p=>p.Value,StringComparer.Ordinal);int count=reader.ReadInt32();if(count!=expected.Count)throw new InvalidDataException("v18 tensor parameter count mismatch");for(int p=0;p<count;p++){string name=reader.ReadString();int length=reader.ReadInt32();if(!expected.TryGetValue(name,out float[] values)||values.Length!=length)throw new InvalidDataException("v18 tensor shape mismatch: "+name);for(int i=0;i<length;i++)values[i]=reader.ReadSingle();}}
    private static void WriteOptimizer(BinaryWriter writer,NativeOptimizerState optimizer,IEnumerable<KeyValuePair<string,float[]>> parameters){optimizer=optimizer??new NativeOptimizerState();writer.Write(optimizer.step);var list=parameters.ToList();writer.Write(list.Count);foreach(var parameter in list){NativeOptimizerSlot slot=optimizer.Get(parameter.Key,parameter.Value.Length);writer.Write(parameter.Key);writer.Write(parameter.Value.Length);foreach(float value in slot.first)writer.Write(value);foreach(float value in slot.second)writer.Write(value);}}
    private static void ReadOptimizer(BinaryReader reader,NativeOptimizerState optimizer,IEnumerable<KeyValuePair<string,float[]>> parameters){optimizer.step=reader.ReadInt32();var expected=parameters.ToDictionary(p=>p.Key,p=>p.Value.Length,StringComparer.Ordinal);int count=reader.ReadInt32();if(count!=expected.Count)throw new InvalidDataException("v18 optimizer slot count mismatch");optimizer.slots=new List<NativeOptimizerSlot>();for(int p=0;p<count;p++){string name=reader.ReadString();int length=reader.ReadInt32();if(!expected.TryGetValue(name,out int expectedLength)||length!=expectedLength)throw new InvalidDataException("v18 optimizer shape mismatch: "+name);NativeOptimizerSlot slot=optimizer.Get(name,length);for(int i=0;i<length;i++)slot.first[i]=reader.ReadSingle();for(int i=0;i<length;i++)slot.second[i]=reader.ReadSingle();}}
}
