# Homeguard Trading Bot - AWS Infrastructure
# Terraform configuration for deploying the trading bot on EC2

terraform {
  required_version = ">= 1.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

# Provider configuration
provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "Homeguard"
      Environment = var.environment
      ManagedBy   = "Terraform"
    }
  }
}

# Data source: Get latest Amazon Linux 2023 AMI (ARM64)
data "aws_ami" "amazon_linux_2023_arm64" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["al2023-ami-2023.*-arm64"]
  }

  filter {
    name   = "architecture"
    values = ["arm64"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

# Security Group - SSH access only
resource "aws_security_group" "homeguard_trading" {
  name        = "homeguard-trading-bot-sg"
  description = "Security group for Homeguard trading bot - SSH access only"

  # Ingress: SSH from specified CIDR
  ingress {
    description = "SSH access"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = var.ssh_allowed_cidrs
  }

  # Egress: Allow all outbound (needed for Alpaca API, yum updates, etc.)
  egress {
    description = "Allow all outbound"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "homeguard-trading-bot-sg"
  }
}

# EC2 Instance - Trading Bot
resource "aws_instance" "homeguard_trading" {
  ami           = data.aws_ami.amazon_linux_2023_arm64.id
  instance_type = var.instance_type
  key_name      = var.key_pair_name

  vpc_security_group_ids = [aws_security_group.homeguard_trading.id]

  # Root volume configuration
  root_block_device {
    volume_type           = "gp3"
    volume_size           = var.root_volume_size
    delete_on_termination = var.delete_volume_on_termination
    encrypted             = true

    tags = {
      Name = "homeguard-trading-bot-root"
    }
  }

  # User data script - automated setup
  user_data = templatefile("${path.module}/user-data.sh", {
    git_repo_url    = var.git_repo_url
    git_branch      = var.git_branch
    alpaca_key_id   = var.alpaca_key_id
    alpaca_secret   = var.alpaca_secret
  })

  # Enable detailed monitoring (costs $2.10/month extra, optional)
  monitoring = var.enable_detailed_monitoring

  # Metadata options (IMDSv2 required for security)
  metadata_options {
    http_endpoint               = "enabled"
    http_tokens                 = "required"  # Require IMDSv2
    http_put_response_hop_limit = 1
  }

  tags = {
    Name = "homeguard-trading-bot"
  }

  lifecycle {
    ignore_changes = [
      ami,  # Don't recreate if AMI updates
      user_data  # Don't recreate if user data changes
    ]
  }
}

# Elastic IP (optional - provides static IP)
resource "aws_eip" "homeguard_trading" {
  count = var.create_elastic_ip ? 1 : 0

  instance = aws_instance.homeguard_trading.id
  domain   = "vpc"

  tags = {
    Name = "homeguard-trading-bot-eip"
  }

  depends_on = [aws_instance.homeguard_trading]
}

# CloudWatch Log Group (optional - if you want to add CloudWatch logging later)
resource "aws_cloudwatch_log_group" "trading_logs" {
  count = var.create_cloudwatch_logs ? 1 : 0

  name              = "/aws/trading-bot/homeguard"
  retention_in_days = var.log_retention_days

  tags = {
    Name = "homeguard-trading-logs"
  }
}

# SNS Topic for Alerts (optional)
resource "aws_sns_topic" "trading_alerts" {
  count = var.create_sns_alerts ? 1 : 0

  name = "homeguard-trading-alerts"

  tags = {
    Name = "homeguard-trading-alerts"
  }
}

resource "aws_sns_topic_subscription" "trading_alerts_email" {
  count = var.create_sns_alerts && var.alert_email != "" ? 1 : 0

  topic_arn = aws_sns_topic.trading_alerts[0].arn
  protocol  = "email"
  endpoint  = var.alert_email
}

# CloudWatch Alarm - Instance Status Check (optional)
resource "aws_cloudwatch_metric_alarm" "instance_status_check" {
  count = var.create_cloudwatch_alarms ? 1 : 0

  alarm_name          = "homeguard-trading-bot-status-check"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "StatusCheckFailed"
  namespace           = "AWS/EC2"
  period              = 60
  statistic           = "Average"
  threshold           = 0
  alarm_description   = "Alert when EC2 instance fails status checks"
  alarm_actions       = var.create_sns_alerts ? [aws_sns_topic.trading_alerts[0].arn] : []

  dimensions = {
    InstanceId = aws_instance.homeguard_trading.id
  }
}
