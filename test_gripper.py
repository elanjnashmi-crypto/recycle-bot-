#!/usr/bin/env python3
import roslibpy
import time

def call_service(client, service_name):
    """Call a std_srvs/Trigger service and print the response."""
    service = roslibpy.Service(client, service_name, 'std_srvs/Trigger')
    request = roslibpy.ServiceRequest({})  # Trigger service takes no arguments

    print(f"[ROS] Calling service: {service_name}")
    result = service.call(request)
    print(f"[ROS] Response: success={result['success']}, message='{result['message']}'")

def main():
    ros_host = '192.168.27.1'  # Replace with your ROS bridge IP
    ros_port = 9090            # rosbridge_websocket port

    client = roslibpy.Ros(host=ros_host, port=ros_port)
    try:
        client.run()
        if not client.is_connected:
            raise RuntimeError("Failed to connect to rosbridge.")

        print("[ROS] Connected to rosbridge.")

        # Open gripper
        call_service(client, '/onrobot/open')
        time.sleep(2)

        # Close gripper
        call_service(client, '/onrobot/close')
        time.sleep(2)

    finally:
        client.terminate()
        print("[ROS] Disconnected.")

if __name__ == '__main__':
    main()
