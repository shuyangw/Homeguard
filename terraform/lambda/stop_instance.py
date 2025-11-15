"""
Lambda Function: Stop EC2 Instance

Stops the Homeguard trading bot EC2 instance after market close.
Triggered by EventBridge rule at 4:30 PM ET on weekdays.
"""

import os
import boto3
import json
from datetime import datetime

# Initialize EC2 client
ec2 = boto3.client('ec2')

def handler(event, context):
    """
    Lambda handler to stop EC2 instance.

    Args:
        event: EventBridge event (contains schedule info)
        context: Lambda context

    Returns:
        Response with status code and message
    """

    instance_id = os.environ['INSTANCE_ID']

    print(f"Stopping instance {instance_id} at {datetime.utcnow().isoformat()}")

    try:
        # Check current instance state
        response = ec2.describe_instances(InstanceIds=[instance_id])
        instance_state = response['Reservations'][0]['Instances'][0]['State']['Name']

        print(f"Current instance state: {instance_state}")

        # Only stop if instance is running
        if instance_state == 'running':
            print(f"Stopping instance {instance_id}...")
            ec2.stop_instances(InstanceIds=[instance_id])

            return {
                'statusCode': 200,
                'body': json.dumps({
                    'message': f'Successfully stopped instance {instance_id}',
                    'instance_id': instance_id,
                    'previous_state': instance_state,
                    'timestamp': datetime.utcnow().isoformat()
                })
            }
        elif instance_state == 'stopped':
            print(f"Instance {instance_id} is already stopped")
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'message': f'Instance {instance_id} is already stopped',
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
                    'message': f'Cannot stop instance in state: {instance_state}',
                    'instance_id': instance_id,
                    'state': instance_state,
                    'timestamp': datetime.utcnow().isoformat()
                })
            }

    except Exception as e:
        print(f"Error stopping instance {instance_id}: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'message': f'Error stopping instance: {str(e)}',
                'instance_id': instance_id,
                'timestamp': datetime.utcnow().isoformat()
            })
        }
