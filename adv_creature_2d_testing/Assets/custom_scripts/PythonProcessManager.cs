using System;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.IO;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using Newtonsoft.Json.Linq;
using UnityEngine;
using Debug = UnityEngine.Debug;

[DefaultExecutionOrder(-100)]
public class PythonProcessManager : MonoBehaviour
{
    public PythonBridge bridge;
    public string Status { get; private set; } = "Starting Python";
    public string LastError { get; private set; } = "";
    public bool OwnsProcess => ownedProcess != null && !ownedProcess.HasExited;
    private Process ownedProcess;
    private bool shuttingDown;
    private readonly ConcurrentQueue<ProbeResult> probeResults = new ConcurrentQueue<ProbeResult>();

    private void Start()
    {
#if UNITY_EDITOR || DEVELOPMENT_BUILD
        int requestedPort = bridge == null ? 9999 : bridge.port;
        string project = Directory.GetParent(Application.dataPath).FullName;
        SetStatus("Checking Python server asynchronously");
        ThreadPool.QueueUserWorkItem(_ => ProbeServer(project, requestedPort));
#else
        SetStatus("Waiting for external Python server");
#endif
    }

    private void ProbeServer(string project, int requestedPort)
    {
        try {
            if (PortIsCompatible(requestedPort)) { probeResults.Enqueue(new ProbeResult(project, requestedPort, true, "Using existing Python server")); return; }
            int selected = PortIsOpen(requestedPort) ? FindFreeFallbackPort() : requestedPort;
            probeResults.Enqueue(new ProbeResult(project, selected, false,
                selected == requestedPort ? "Starting project Python" : "Incompatible server detected; using port " + selected));
        } catch (Exception exception) { probeResults.Enqueue(new ProbeResult(project, requestedPort, false, "Python probe failed: " + exception.Message)); }
    }

    private bool PortIsOpen(int port)
    {
        try {
            using (TcpClient client = new TcpClient()) {
                client.Connect("127.0.0.1", port);
                return client.Connected;
            }
        } catch { return false; }
    }

    private bool PortIsCompatible(int port)
    {
        try {
            using (TcpClient client = new TcpClient()) {
                client.Connect("127.0.0.1", port);
                if (!client.Connected) return false;
                client.ReceiveTimeout = 1000;
                NetworkStream stream = client.GetStream(); byte[] header = ReadExact(stream, 4);
                if (BitConverter.IsLittleEndian) Array.Reverse(header);
                int size = BitConverter.ToInt32(header, 0);
                if (size <= 0 || size > 8 * 1024 * 1024) return false;
                JObject response = JObject.Parse(Encoding.UTF8.GetString(ReadExact(stream, size)));
                return response.Value<int?>("protocol_version") == PythonBridge.ProtocolVersion;
            }
        } catch { return false; }
    }

    private static byte[] ReadExact(NetworkStream stream, int count)
    {
        byte[] data = new byte[count]; int offset = 0;
        while (offset < count) { int read = stream.Read(data, offset, count - offset); if (read <= 0) throw new IOException("server closed probe"); offset += read; }
        return data;
    }

    private int FindFreeFallbackPort()
    {
        for (int candidate = 10000; candidate < 10010; candidate++) if (!PortIsOpen(candidate)) return candidate;
        throw new InvalidOperationException("No free fallback ecosystem port was found");
    }

    private void TryStartProjectPython(string project, int selectedPort)
    {
        try {
            string repository = Directory.GetParent(project).FullName;
            string python = Path.Combine(repository, ".venv", "Scripts", "python.exe");
            string server = Path.Combine(project, "python", "server.py");
            if (!File.Exists(python)) throw new FileNotFoundException("Project venv Python was not found", python);
            if (!File.Exists(server)) throw new FileNotFoundException("Python ecosystem server was not found", server);
            ProcessStartInfo start = new ProcessStartInfo {
                FileName = python,
                Arguments = "-u \"" + server + "\"",
                WorkingDirectory = Path.GetDirectoryName(server),
                UseShellExecute = false,
                CreateNoWindow = true,
                RedirectStandardOutput = false,
                RedirectStandardError = false
            };
            start.EnvironmentVariables["LIVE_ECOSYSTEM_PORT"] = selectedPort.ToString();
            ownedProcess = new Process { StartInfo = start, EnableRaisingEvents = false };
            if (!ownedProcess.Start()) throw new InvalidOperationException("Python process did not start");
            SetStatus("Python running");
        } catch (Exception exception) {
            LastError = exception.Message; SetStatus("Python start failed"); Debug.LogError("Python auto-start failed: " + exception);
        }
    }

    private void Update()
    {
        while (probeResults.TryDequeue(out ProbeResult result)) {
            if (result.usesExisting) { SetStatus(result.status); continue; }
            if (bridge != null) bridge.port = result.port;
            SetStatus(result.status);
            TryStartProjectPython(result.project, result.port);
        }
        if (!shuttingDown && ownedProcess != null && ownedProcess.HasExited) SetStatus("Python exited");
    }

    private void SetStatus(string value)
    {
        if (Status == value) return;
        Status = value; Debug.Log("Python process: " + value);
    }

    private void StopOwnedProcess()
    {
        if (shuttingDown) return;
        shuttingDown = true;
        if (ownedProcess == null) return;
        try {
            if (!ownedProcess.HasExited) {
                if (bridge != null) bridge.RequestOwnedServerShutdown();
                if (!ownedProcess.WaitForExit(1500)) ownedProcess.Kill();
            }
        } catch (Exception exception) { Debug.LogWarning("Could not stop owned Python process: " + exception.Message); }
        finally { ownedProcess.Dispose(); ownedProcess = null; }
    }

    private void OnApplicationQuit() { StopOwnedProcess(); }
    private void OnDestroy() { StopOwnedProcess(); }

    private readonly struct ProbeResult
    {
        public readonly string project, status;
        public readonly int port;
        public readonly bool usesExisting;
        public ProbeResult(string project, int port, bool usesExisting, string status)
        { this.project = project; this.port = port; this.usesExisting = usesExisting; this.status = status; }
    }
}
