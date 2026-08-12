import socket
import struct
import json
import torch
import math
import os
import signal
import sys
from typing import Dict, List, Any

from creature_brain_m123 import CreatureBrainM123
from replay_buffer import ReplayBuffer

HOST = "127.0.0.1"
PORT = 9999
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "checkpoints", "checkpoint_fixed_creature_0.pt")

# Global Brain and Replay Buffer instances
brain = CreatureBrainM123(catalog_size=64, d_model=64, inner_loop_steps=3, inner_lr=0.01, m3_lr=0.001)
replay_buffer = ReplayBuffer(capacity=5000)

# Attempt to load existing checkpoint from previous runs
brain.load_checkpoint(CHECKPOINT_PATH)

# Memory for tracking creature state across ticks
creature_prev_state: Dict[str, Dict[str, Any]] = {}
tick_counter = 0

def recv_exact(conn, n):
    data = b""
    while len(data) < n:
        chunk = conn.recv(n - len(data))
        if not chunk:
            raise ConnectionError("Socket closed while reading")
        data += chunk
    return data

def recv_message(conn):
    length_bytes = recv_exact(conn, 4)
    (length,) = struct.unpack(">I", length_bytes)
    payload = recv_exact(conn, length)
    return json.loads(payload.decode("utf-8"))

def send_message(conn, obj):
    payload = json.dumps(obj).encode("utf-8")
    header = struct.pack(">I", len(payload))
    conn.sendall(header + payload)

def compute_outputs(inputs_message):
    global tick_counter
    tick_counter += 1
    response = {"creatures": {}}
    
    for creature_id, creature_data in inputs_message["creatures"].items():
        local_inputs = creature_data.get("local_inputs", {})
        global_inputs = creature_data.get("global_inputs", {})
        goal_setter_inputs = creature_data.get("goal_setter_inputs", [])
        
        limb_ids = list(local_inputs.keys())
        if not limb_ids:
            response["creatures"][creature_id] = {"deltas": {}, "telemetry": {}}
            continue

        # Convert string limb IDs to numeric catalog indices (1..catalog_size-1)
        joint_catalog_ids = [ (abs(hash(lid)) % 60) + 1 for lid in limb_ids ]
        
        # Extract 4D joint sensor data per limb: [joint_angle, touch_self, touch_other, touch_env]
        joint_sensor_data = []
        for lid in limb_ids:
            l_dict = local_inputs[lid]
            angle = float(l_dict.get("joint_angle", 0.0))
            ts = float(l_dict.get("touch_self", 0.0))
            to = float(l_dict.get("touch_other_creature", 0.0))
            te = float(l_dict.get("touch_environment", 0.0))
            joint_sensor_data.append([angle, ts, to, te])
            
        pos_x = float(global_inputs.get("position_x", 0.0))
        pos_y = float(global_inputs.get("position_y", 0.0))

        # ---------------------------------------------------------------------
        # M1 NEAT Goal-Setter: Compute goal vector g_t purely from NEAT network
        # processing the 5-frame memory history vector (no manual hardcoding!)
        # ---------------------------------------------------------------------
        goal_tensor = brain.m1.forward(goal_setter_inputs)
        goal_x, goal_y, goal_z = goal_tensor[0].item(), goal_tensor[1].item(), goal_tensor[2].item()
        goal_source = f"M1 NEAT Goal-Setter (Input dim: {len(goal_setter_inputs)})"

        m3_loss_val = 0.0
        m3_train_telem = {}
        
        # 1. Supervised M3 Dynamics Update from previous frame transition
        if creature_id in creature_prev_state:
            prev = creature_prev_state[creature_id]
            real_delta_x = pos_x - prev["pos_x"]
            real_delta_y = pos_y - prev["pos_y"]
            real_delta_pos = torch.tensor([real_delta_x, real_delta_y, 0.0], dtype=torch.float32)
            
            # Train M3 supervised update
            m3_loss_val, m3_train_telem = brain.update_m3_dynamics(prev["joint_ids"], prev["actions"], real_delta_pos)
            
            # Push transition into experience replay buffer
            replay_buffer.push(
                joint_ids=prev["joint_ids"],
                joint_angles=[s[0] for s in prev["joint_sensor_data"]],
                goal=prev["goal"],
                action=prev["actions"],
                real_delta_pos=real_delta_pos
            )
            
        # 2. Select actions via M2 inner-loop backprop through M3
        actions_tensor, m2_m3_telem = brain.select_action(
            joint_catalog_ids,
            joint_sensor_data,
            goal_tensor,
            inner_steps=3
        )
        
        action_list = actions_tensor.tolist()
        deltas = { limb_ids[i]: float(action_list[i]) for i in range(len(limb_ids)) }
        
        # Store state for next tick's transition
        creature_prev_state[creature_id] = {
            "joint_ids": joint_catalog_ids,
            "joint_sensor_data": joint_sensor_data,
            "goal": goal_tensor,
            "actions": actions_tensor,
            "pos_x": pos_x,
            "pos_y": pos_y
        }

        # Build Telemetry Dictionary for Unity HUD and Python terminal printing
        telemetry = {
            "active_phase": "M1 (NEAT Goal-Setter) -> M2 (Action Selector) -> M3 (Dynamics Predictor)",
            "goal_source": goal_source,
            "relative_goal": [round(goal_x, 3), round(goal_y, 3), round(goal_z, 3)],
            "m2_joint_sensors": [[round(v, 2) for v in s] for s in joint_sensor_data],
            "m2_output_deltas": [round(d, 2) for d in action_list],
            "m3_pred_displacement": [round(x, 4) for x in m2_m3_telem["m3_pred_displacement"]],
            "m3_goal_loss": round(m2_m3_telem["m3_goal_loss"], 5),
            "m3_real_displacement": [round(x, 4) for x in m3_train_telem.get("m3_real_displacement", [0.0, 0.0, 0.0])],
            "m3_dynamics_loss": round(m3_loss_val, 5),
            "replay_buffer_size": len(replay_buffer)
        }

        response["creatures"][creature_id] = {
            "deltas": deltas,
            "telemetry": telemetry
        }

        # Save checkpoint periodically every 100 ticks
        if tick_counter % 100 == 0:
            brain.save_checkpoint(CHECKPOINT_PATH)

        # Detailed Terminal Telemetry Output (printed every 50 ticks)
        if tick_counter % 50 == 0:
            print("\n" + "=" * 80)
            print(f" [BRAIN TELEMETRY | Tick #{tick_counter} | Creature: {creature_id}]")
            print(" " + "-" * 78)
            print(f" [ACTIVE NETWORKS]  : {telemetry['active_phase']}")
            print(f" [1. M1 NEAT GOAL]  : {goal_source} => Relative Goal g_t = [{goal_x:+.3f}, {goal_y:+.3f}, {goal_z:+.3f}]")
            print(f" [2. M2 ACHIEVER]   : Input 4D Tokens [Angle, TouchS, TouchO, TouchE] = {telemetry['m2_joint_sensors']}")
            print(f"                    : Output Joint Deltas = {telemetry['m2_output_deltas']}")
            print(f" [3. M3 ACHIEVER]   : Pred Δp={telemetry['m3_pred_displacement']} | Goal MSE Loss={telemetry['m3_goal_loss']:.5f}")
            print(f" [4. M3 TRAIN]      : Real Δp={telemetry['m3_real_displacement']} | Dynamics MSE Loss={telemetry['m3_dynamics_loss']:.5f}")
            print(f" [5. REPLAY BUFFER] : {telemetry['replay_buffer_size']} transitions stored")
            print("=" * 80 + "\n")
        
    return response

def handle_exit_signal(sig, frame):
    print("\n[M123 Brain Server] Shutdown signal received. Saving checkpoint before exiting...")
    brain.save_checkpoint(CHECKPOINT_PATH)
    sys.exit(0)

def main():
    signal.signal(signal.SIGINT, handle_exit_signal)
    signal.signal(signal.SIGTERM, handle_exit_signal)

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen(1)
    print(f"[M123 Brain Server] Telemetry & NEAT M1 active. Waiting for Unity connection on {HOST}:{PORT}...")
    conn, addr = server.accept()
    print(f"[M123 Brain Server] Unity connected from {addr}")

    try:
        while True:
            inputs_message = recv_message(conn)
            outputs_message = compute_outputs(inputs_message)
            send_message(conn, outputs_message)
    except ConnectionError:
        print("[M123 Brain Server] Unity disconnected. Saving final checkpoint...")
        brain.save_checkpoint(CHECKPOINT_PATH)
    finally:
        conn.close()
        server.close()

if __name__ == "__main__":
    main()