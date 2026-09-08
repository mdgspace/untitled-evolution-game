using System;
using System.Collections.Generic;
using System.Linq;

[Serializable] public sealed class GoalNode{public int id;public float depth;public GoalNode Copy()=>new GoalNode{id=id,depth=depth};}
[Serializable] public sealed class GoalConnection{public int innovation,source,target;public float weight;public bool enabled=true;public GoalConnection Copy()=>(GoalConnection)MemberwiseClone();}
[Serializable] public sealed class GoalInnovation{public int source,target,innovation,splitNode=-1;}
[Serializable]
public sealed class RtNeatGoalGenome
{
    public List<GoalNode> nodes=new List<GoalNode>();public List<GoalConnection> connections=new List<GoalConnection>();public float fitness;public int age,species;
    public RtNeatGoalGenome Copy(){var copy=new RtNeatGoalGenome{fitness=fitness,age=age,species=species};foreach(var node in nodes)copy.nodes.Add(node.Copy());foreach(var connection in connections)copy.connections.Add(connection.Copy());return copy;}
}

// Population-owned innovation history makes structural mutations homologous at crossover.
// Phenotypes stay acyclic and feed-forward so evaluating M1 does not allocate per tick.
[Serializable]
public sealed class NativeRtNeatGoalPolicy
{
    public const int InputCount=28,BiasNode=28,FirstOutput=29,OutputCount=4,MaximumNodes=97,MaximumConnections=1024;
    public List<GoalInnovation> innovations=new List<GoalInnovation>();public int nextNode=33,nextInnovation=1;public NativeDeterministicRng random=new NativeDeterministicRng(3571);
    private int NextInt(int count)=>Math.Min(count-1,(int)(random.NextFloat()*count));
    private float Normal(){float u=Math.Max(1e-7f,random.NextFloat()),v=random.NextFloat();return (float)(Math.Sqrt(-2d*Math.Log(u))*Math.Cos(2d*Math.PI*v));}
    public int Innovation(int source,int target){foreach(var item in innovations)if(item.source==source&&item.target==target)return item.innovation;int id=nextInnovation++;innovations.Add(new GoalInnovation{source=source,target=target,innovation=id});return id;}
    public RtNeatGoalGenome Seed()
    {
        var genome=new RtNeatGoalGenome();for(int i=0;i<FirstOutput+OutputCount;i++)genome.nodes.Add(new GoalNode{id=i,depth=i<=BiasNode?0f:1f});
        for(int i=0;i<=BiasNode;i++)for(int output=FirstOutput;output<FirstOutput+OutputCount;output++)genome.connections.Add(new GoalConnection{innovation=Innovation(i,output),source=i,target=output,weight=(random.NextFloat()*2f-1f)*.3f});return genome;
    }
    public RtNeatGoalGenome Breed(RtNeatGoalGenome best,RtNeatGoalGenome other)
    {
        if(best==null)return Seed();if(other==null)other=best;var child=best.Copy();child.age=0;child.fitness=0f;
        foreach(var connection in child.connections){GoalConnection match=other.connections.FirstOrDefault(c=>c.innovation==connection.innovation);if(match!=null){if(random.NextFloat()<.5f)connection.weight=match.weight;if(!connection.enabled||!match.enabled)connection.enabled=random.NextFloat()<.25f;}if(random.NextFloat()<.8f)connection.weight=Math.Max(-5f,Math.Min(5f,connection.weight+Normal()*.12f));}
        if(random.NextFloat()<.15f&&child.nodes.Count<MaximumNodes)Split(child);if(random.NextFloat()<.3f&&Enabled(child)<MaximumConnections)Connect(child);return child;
    }
    public void Split(RtNeatGoalGenome genome)
    {
        if(genome.nodes.Count>=MaximumNodes||Enabled(genome)>=MaximumConnections-1)return;var candidates=genome.connections.Where(c=>c.enabled).ToList();if(candidates.Count==0)return;GoalConnection edge=candidates[NextInt(candidates.Count)];GoalInnovation history=innovations.FirstOrDefault(i=>i.innovation==edge.innovation);if(history==null)throw new InvalidOperationException("Missing rtNEAT innovation history");if(history.splitNode<0)history.splitNode=nextNode++;if(genome.nodes.Any(n=>n.id==history.splitNode))return;GoalNode source=genome.nodes.First(n=>n.id==edge.source),target=genome.nodes.First(n=>n.id==edge.target);float depth=(source.depth+target.depth)*.5f;if(depth<=source.depth||depth>=target.depth)return;edge.enabled=false;genome.nodes.Add(new GoalNode{id=history.splitNode,depth=depth});genome.connections.Add(new GoalConnection{innovation=Innovation(source.id,history.splitNode),source=source.id,target=history.splitNode,weight=1f});genome.connections.Add(new GoalConnection{innovation=Innovation(history.splitNode,target.id),source=history.splitNode,target=target.id,weight=edge.weight});
    }
    private void Connect(RtNeatGoalGenome genome){for(int retry=0;retry<32;retry++){GoalNode a=genome.nodes[NextInt(genome.nodes.Count)],b=genome.nodes[NextInt(genome.nodes.Count)];if(a.depth>=b.depth)continue;GoalConnection old=genome.connections.FirstOrDefault(c=>c.source==a.id&&c.target==b.id);if(old!=null){old.enabled=true;return;}genome.connections.Add(new GoalConnection{innovation=Innovation(a.id,b.id),source=a.id,target=b.id,weight=Normal()*.3f});return;}}
    public static int Enabled(RtNeatGoalGenome genome)=>genome?.connections?.Count(c=>c.enabled)??0;
    public static float Distance(RtNeatGoalGenome a,RtNeatGoalGenome b){int unmatched=0,matched=0;float delta=0f;foreach(var c in a.connections){GoalConnection d=b.connections.FirstOrDefault(x=>x.innovation==c.innovation);if(d==null)unmatched++;else{matched++;delta+=Math.Abs(c.weight-d.weight);}}foreach(var c in b.connections)if(!a.connections.Any(x=>x.innovation==c.innovation))unmatched++;return unmatched/(float)Math.Max(1,Math.Max(a.connections.Count,b.connections.Count))+.4f*delta/Math.Max(1,matched);}
    public static void Speciate(IReadOnlyList<RtNeatGoalGenome> genomes){var representatives=new List<RtNeatGoalGenome>();foreach(var genome in genomes){int index=representatives.FindIndex(r=>Distance(genome,r)<.45f);if(index<0){index=representatives.Count;representatives.Add(genome);}genome.species=index;}}
}

public sealed class RtNeatGoalPhenotype
{
    private readonly float[] values;private readonly int[] nodeIds,outputs=new int[NativeRtNeatGoalPolicy.OutputCount];private readonly GoalConnection[][] incoming;
    public RtNeatGoalPhenotype(RtNeatGoalGenome genome)
    {
        var nodes=genome.nodes.OrderBy(n=>n.depth).ThenBy(n=>n.id).ToList();values=new float[nodes.Count];nodeIds=nodes.Select(n=>n.id).ToArray();incoming=new GoalConnection[nodes.Count][];
        for(int i=0;i<nodes.Count;i++){if(nodes[i].id>=NativeRtNeatGoalPolicy.FirstOutput&&nodes[i].id<NativeRtNeatGoalPolicy.FirstOutput+NativeRtNeatGoalPolicy.OutputCount)outputs[nodes[i].id-NativeRtNeatGoalPolicy.FirstOutput]=i;var edges=new List<GoalConnection>();foreach(var source in genome.connections)if(source.enabled&&source.target==nodes[i].id){var edge=source.Copy();edge.source=nodes.FindIndex(n=>n.id==source.source);if(edge.source<0||edge.source>=i)throw new InvalidOperationException("rtNEAT graph is not acyclic");edges.Add(edge);}incoming[i]=edges.ToArray();}
    }
    public void Evaluate(float[] inputs,float[] output)
    {
        if(inputs==null||inputs.Length<NativeRtNeatGoalPolicy.InputCount||output==null||output.Length<NativeRtNeatGoalPolicy.OutputCount)throw new ArgumentException("rtNEAT M1 input/output shape mismatch");
        for(int i=0;i<values.Length;i++){int id=nodeIds[i];if(id<NativeRtNeatGoalPolicy.InputCount){values[i]=inputs[id];continue;}if(id==NativeRtNeatGoalPolicy.BiasNode){values[i]=1f;continue;}float sum=0f;foreach(GoalConnection edge in incoming[i])sum+=values[edge.source]*edge.weight;values[i]=(float)Math.Tanh(sum);}for(int i=0;i<outputs.Length;i++)output[i]=values[outputs[i]];
    }
}
