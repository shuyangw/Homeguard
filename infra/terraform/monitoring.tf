# CloudWatch Agent monitoring for EC2 instance
# Enables memory/disk metrics (not available by default in EC2)

# IAM role for the EC2 instance to push CloudWatch metrics
resource "aws_iam_role" "ec2_cloudwatch" {
  count = var.enable_cloudwatch_agent ? 1 : 0

  name = "homeguard-ec2-cloudwatch"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })

  tags = {
    Name = "homeguard-ec2-cloudwatch"
  }
}

resource "aws_iam_role_policy_attachment" "cloudwatch_agent" {
  count = var.enable_cloudwatch_agent ? 1 : 0

  role       = aws_iam_role.ec2_cloudwatch[0].name
  policy_arn = "arn:aws:iam::aws:policy/CloudWatchAgentServerPolicy"
}

resource "aws_iam_role_policy_attachment" "ssm_core" {
  count = var.enable_cloudwatch_agent ? 1 : 0

  role       = aws_iam_role.ec2_cloudwatch[0].name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_instance_profile" "ec2_cloudwatch" {
  count = var.enable_cloudwatch_agent ? 1 : 0

  name = "homeguard-ec2-cloudwatch"
  role = aws_iam_role.ec2_cloudwatch[0].name
}

# CloudWatch alarm: high memory usage
resource "aws_cloudwatch_metric_alarm" "high_memory" {
  count = var.enable_cloudwatch_agent ? 1 : 0

  alarm_name          = "homeguard-high-memory-usage"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "mem_used_percent"
  namespace           = "CWAgent"
  period              = 300
  statistic           = "Average"
  threshold           = 85
  alarm_description   = "EC2 memory usage above 85% for 10 minutes"
  alarm_actions       = var.create_sns_alerts ? [aws_sns_topic.trading_alerts[0].arn] : []

  dimensions = {
    InstanceId = aws_instance.homeguard_trading.id
  }
}

# CloudWatch alarm: swap usage (indicates memory pressure)
resource "aws_cloudwatch_metric_alarm" "swap_usage" {
  count = var.enable_cloudwatch_agent ? 1 : 0

  alarm_name          = "homeguard-swap-usage"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "swap_used_percent"
  namespace           = "CWAgent"
  period              = 300
  statistic           = "Average"
  threshold           = 50
  alarm_description   = "EC2 swap usage above 50% - memory pressure"
  alarm_actions       = var.create_sns_alerts ? [aws_sns_topic.trading_alerts[0].arn] : []

  dimensions = {
    InstanceId = aws_instance.homeguard_trading.id
  }
}
