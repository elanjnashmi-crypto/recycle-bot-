"""
Opens the OnRobot gripper and nothing else (the arm does not move).
Keep fingers and objects clear of the jaws. Run:  python gripper_open_only.py
Add  close  to close it instead:  python gripper_open_only.py close
"""
import sys
import time
import roslibpy

action = sys.argv[1] if len(sys.argv) > 1 else "open"
service_name = "/onrobot/close" if action == "close" else "/onrobot/open"

client = roslibpy.Ros(host="192.168.27.1", port=9090)
try:
    client.run()
    service = roslibpy.Service(client, service_name, "std_srvs/Trigger")
    result = service.call(roslibpy.ServiceRequest({}), timeout=10)
    print(f"{service_name}: success={result['success']} message='{result['message']}'")
    time.sleep(2)
finally:
    client.terminate()