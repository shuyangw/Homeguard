# Scheduled Start/Stop for EC2 Instance
# Automatically starts instance before market open, stops after market close
# Saves ~70% on EC2 costs (but Elastic IP required)

# IAM Role for Lambda to start/stop EC2 instances
resource "aws_iam_role" "ec2_scheduler" {
  count = var.enable_scheduled_start_stop ? 1 : 0

  name = "homeguard-ec2-scheduler-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "lambda.amazonaws.com"
        }
      }
    ]
  })

  tags = {
    Name = "homeguard-ec2-scheduler-role"
  }
}

# IAM Policy for Lambda to manage EC2 instances
resource "aws_iam_role_policy" "ec2_scheduler_policy" {
  count = var.enable_scheduled_start_stop ? 1 : 0

  name = "homeguard-ec2-scheduler-policy"
  role = aws_iam_role.ec2_scheduler[0].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "ec2:StartInstances",
          "ec2:StopInstances",
          "ec2:DescribeInstances"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "arn:aws:logs:*:*:*"
      }
    ]
  })
}

# Lambda Function to START instance
resource "aws_lambda_function" "start_instance" {
  count = var.enable_scheduled_start_stop ? 1 : 0

  filename      = "${path.module}/lambda_start_instance.zip"
  function_name = "homeguard-start-instance"
  role          = aws_iam_role.ec2_scheduler[0].arn
  handler       = "index.handler"
  runtime       = "python3.11"
  timeout       = 60

  environment {
    variables = {
      INSTANCE_ID = aws_instance.homeguard_trading.id
    }
  }

  tags = {
    Name = "homeguard-start-instance"
  }

  depends_on = [
    aws_iam_role_policy.ec2_scheduler_policy
  ]
}

# Lambda Function to STOP instance
resource "aws_lambda_function" "stop_instance" {
  count = var.enable_scheduled_start_stop ? 1 : 0

  filename      = "${path.module}/lambda_stop_instance.zip"
  function_name = "homeguard-stop-instance"
  role          = aws_iam_role.ec2_scheduler[0].arn
  handler       = "index.handler"
  runtime       = "python3.11"
  timeout       = 60

  environment {
    variables = {
      INSTANCE_ID = aws_instance.homeguard_trading.id
    }
  }

  tags = {
    Name = "homeguard-stop-instance"
  }

  depends_on = [
    aws_iam_role_policy.ec2_scheduler_policy
  ]
}

# EventBridge Rule: Start instance at 9:00 AM ET (Monday-Friday)
resource "aws_cloudwatch_event_rule" "start_instance" {
  count = var.enable_scheduled_start_stop ? 1 : 0

  name                = "homeguard-start-instance"
  description         = "Start trading bot instance at 9:00 AM ET on weekdays"
  schedule_expression = "cron(0 14 ? * MON-FRI *)"  # 14:00 UTC = 9:00 AM ET

  tags = {
    Name = "homeguard-start-instance-rule"
  }
}

# EventBridge Rule: Stop instance at 4:30 PM ET (Monday-Friday)
resource "aws_cloudwatch_event_rule" "stop_instance" {
  count = var.enable_scheduled_start_stop ? 1 : 0

  name                = "homeguard-stop-instance"
  description         = "Stop trading bot instance at 4:30 PM ET on weekdays"
  schedule_expression = "cron(30 21 ? * MON-FRI *)"  # 21:30 UTC = 4:30 PM ET

  tags = {
    Name = "homeguard-stop-instance-rule"
  }
}

# EventBridge Target: Start instance
resource "aws_cloudwatch_event_target" "start_instance" {
  count = var.enable_scheduled_start_stop ? 1 : 0

  rule      = aws_cloudwatch_event_rule.start_instance[0].name
  target_id = "StartInstanceLambda"
  arn       = aws_lambda_function.start_instance[0].arn
}

# EventBridge Target: Stop instance
resource "aws_cloudwatch_event_target" "stop_instance" {
  count = var.enable_scheduled_start_stop ? 1 : 0

  rule      = aws_cloudwatch_event_rule.stop_instance[0].name
  target_id = "StopInstanceLambda"
  arn       = aws_lambda_function.stop_instance[0].arn
}

# Lambda Permission: Allow EventBridge to invoke START function
resource "aws_lambda_permission" "allow_eventbridge_start" {
  count = var.enable_scheduled_start_stop ? 1 : 0

  statement_id  = "AllowExecutionFromEventBridge"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.start_instance[0].function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.start_instance[0].arn
}

# Lambda Permission: Allow EventBridge to invoke STOP function
resource "aws_lambda_permission" "allow_eventbridge_stop" {
  count = var.enable_scheduled_start_stop ? 1 : 0

  statement_id  = "AllowExecutionFromEventBridge"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.stop_instance[0].function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.stop_instance[0].arn
}

# CloudWatch Log Group for START Lambda
resource "aws_cloudwatch_log_group" "start_instance_logs" {
  count = var.enable_scheduled_start_stop ? 1 : 0

  name              = "/aws/lambda/homeguard-start-instance"
  retention_in_days = 90  # Keep 3 months of scheduling history

  tags = {
    Name = "homeguard-start-instance-logs"
  }
}

# CloudWatch Log Group for STOP Lambda
resource "aws_cloudwatch_log_group" "stop_instance_logs" {
  count = var.enable_scheduled_start_stop ? 1 : 0

  name              = "/aws/lambda/homeguard-stop-instance"
  retention_in_days = 90  # Keep 3 months of scheduling history

  tags = {
    Name = "homeguard-stop-instance-logs"
  }
}
