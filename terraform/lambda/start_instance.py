"""
Lambda Function: Start EC2 Instance

Starts the Homeguard trading bot EC2 instance before market open.
Triggered by EventBridge rule at 9:00 AM ET on weekdays.
"""

import os
import boto3
import json
from datetime import datetime

# Initialize EC2 client
ec2 = boto3.client('ec2')

def handler(event, context):
    """
    Lambda handler to start EC2 instance.

    Args:
        event: EventBridge event (contains schedule info)
        context: Lambda context

    Returns:
        Response with status code and message
    """

    instance_id = os.environ['INSTANCE_ID']

    print(f"Starting instance {instance_id} at {datetime.utcnow().isoformat()}")

    try:
        # Check current instance state
        response = ec2.describe_instances(InstanceIds=[instance_id])
        instance_state = response['Reservations'][0]['Instances'][0]['State']['Name']

        print(f"Current instance state: {instance_state}")

        # Only start if instance is stopped
        if instance_state == 'stopped':
            print(f"Starting instance {instance_id}...")
            ec2.start_instances(InstanceIds=[instance_id])

            return {
                'statusCode': 200,
                'body': json.dumps({
                    'message': f'Successfully started instance {instance_id}',
                    'instance_id': instance_id,
                    'previous_state': instance_state,
                    'timestamp': datetime.utcnow().isoformat()
                })
            }
        elif instance_state == 'running':
            print(f"Instance {instance_id} is already running")
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'message': f'Instance {instance_id} is already running',
                    'instance_id': instance_id,
                    'state': instance_state,
                    'timestamp': datetime.utcnow().isoformat()
                })
            }
        else:
            print(f"Instance {instance_id} is in state: {instance_state}")
            return {
                'statusCode': 400,
                'body': json.dumps({
                    'message': f'Cannot start instance in state: {instance_state}',
                    'instance_id': instance_id,
                    'state': instance_state,
                    'timestamp': datetime.utcnow().isoformat()
                })
            }

    except Exception as e:
        print(f"Error starting instance {instance_id}: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'message': f'Error starting instance: {str(e)}',
                'instance_id': instance_id,
                'timestamp': datetime.utcnow().isoformat()
            })
        }
