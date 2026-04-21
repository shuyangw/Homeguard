# Scheduled Start/Stop for EC2 Instance
# Uses EventBridge Scheduler (DST-aware) instead of EventBridge Rules (UTC-only).
# Weekday cron runs in America/New_York so the window stays 8:00 AM - 4:30 PM ET
# year-round across DST transitions. Weekend cron stays UTC because the CSCM
# Sunday rebalance ticks at 00:00 UTC (not ET).

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

# IAM Role for EventBridge Scheduler to invoke the Lambdas
resource "aws_iam_role" "scheduler_invoke" {
  count = var.enable_scheduled_start_stop ? 1 : 0

  name = "homeguard-scheduler-invoke-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "scheduler.amazonaws.com"
        }
      }
    ]
  })

  tags = {
    Name = "homeguard-scheduler-invoke-role"
  }
}

resource "aws_iam_role_policy" "scheduler_invoke_policy" {
  count = var.enable_scheduled_start_stop ? 1 : 0

  name = "homeguard-scheduler-invoke-policy"
  role = aws_iam_role.scheduler_invoke[0].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = "lambda:InvokeFunction"
        Resource = [
          aws_lambda_function.start_instance[0].arn,
          aws_lambda_function.stop_instance[0].arn
        ]
      }
    ]
  })
}

# EventBridge Scheduler: START instance at 8:00 AM ET (Monday-Friday)
# 90 min premarket buffer before 9:30 AM ET market open.
# Timezone-aware: stays 8:00 AM ET across DST transitions.
resource "aws_scheduler_schedule" "start_instance_weekday" {
  count = var.enable_scheduled_start_stop ? 1 : 0

  name        = "homeguard-start-instance"
  description = "Start trading bot instance at 8:00 AM ET on weekdays (DST-aware)"

  flexible_time_window {
    mode = "OFF"
  }

  schedule_expression          = "cron(0 8 ? * MON-FRI *)"
  schedule_expression_timezone = "America/New_York"

  target {
    arn      = aws_lambda_function.start_instance[0].arn
    role_arn = aws_iam_role.scheduler_invoke[0].arn
  }
}

# EventBridge Scheduler: STOP instance at 4:30 PM ET (Monday-Friday)
# 30 min after 4:00 PM market close, after RAMP 3:55 PM rebalance completes.
resource "aws_scheduler_schedule" "stop_instance_weekday" {
  count = var.enable_scheduled_start_stop ? 1 : 0

  name        = "homeguard-stop-instance"
  description = "Stop trading bot instance at 4:30 PM ET on weekdays (DST-aware)"

  flexible_time_window {
    mode = "OFF"
  }

  schedule_expression          = "cron(30 16 ? * MON-FRI *)"
  schedule_expression_timezone = "America/New_York"

  target {
    arn      = aws_lambda_function.stop_instance[0].arn
    role_arn = aws_iam_role.scheduler_invoke[0].arn
  }
}

# EventBridge Scheduler: START instance 1 hour before CSCM Sunday rebalance.
# CSCM ticks at Sunday 00:00 UTC (UTC-fixed, not ET) -> keep schedule in UTC.
resource "aws_scheduler_schedule" "start_instance_sunday" {
  count = var.enable_scheduled_start_stop ? 1 : 0

  name        = "homeguard-start-instance-sunday"
  description = "Start instance 1 hour before CSCM Sunday rebalance (UTC)"

  flexible_time_window {
    mode = "OFF"
  }

  schedule_expression          = "cron(0 23 ? * SAT *)"
  schedule_expression_timezone = "UTC"

  target {
    arn      = aws_lambda_function.start_instance[0].arn
    role_arn = aws_iam_role.scheduler_invoke[0].arn
  }
}

# EventBridge Scheduler: STOP instance 10 min after CSCM Sunday rebalance.
resource "aws_scheduler_schedule" "stop_instance_sunday" {
  count = var.enable_scheduled_start_stop ? 1 : 0

  name        = "homeguard-stop-instance-sunday"
  description = "Stop instance 10 min after CSCM Sunday rebalance (UTC)"

  flexible_time_window {
    mode = "OFF"
  }

  schedule_expression          = "cron(10 0 ? * SUN *)"
  schedule_expression_timezone = "UTC"

  target {
    arn      = aws_lambda_function.stop_instance[0].arn
    role_arn = aws_iam_role.scheduler_invoke[0].arn
  }
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
