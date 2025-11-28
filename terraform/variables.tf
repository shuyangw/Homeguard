# Terraform Variables for Homeguard Trading Bot

# ===== REQUIRED VARIABLES =====

variable "aws_region" {
  description = "AWS region to deploy resources"
  type        = string
  default     = "us-east-1"
}

variable "key_pair_name" {
  description = "Name of existing EC2 key pair for SSH access"
  type        = string
  # Must be created in AWS Console before running terraform
}

variable "ssh_allowed_cidrs" {
  description = "CIDR blocks allowed to SSH to the instance (your IP address)"
  type        = list(string)
  default     = ["0.0.0.0/0"]  # WARNING: Change this to your IP for security!

  validation {
    condition     = length(var.ssh_allowed_cidrs) > 0
    error_message = "At least one CIDR block must be specified for SSH access."
  }
}

variable "git_repo_url" {
  description = "Git repository URL for Homeguard codebase"
  type        = string
  default     = "https://github.com/shuyangw/Homeguard.git"
}

variable "git_branch" {
  description = "Git branch to checkout"
  type        = string
  default     = "main"
}

# ===== ALPACA API CREDENTIALS =====

variable "alpaca_key_id" {
  description = "Alpaca API Key ID (paper trading)"
  type        = string
  sensitive   = true
  default     = ""
  # Recommended: Set via environment variable TF_VAR_alpaca_key_id
}

variable "alpaca_secret" {
  description = "Alpaca API Secret Key (paper trading)"
  type        = string
  sensitive   = true
  default     = ""
  # Recommended: Set via environment variable TF_VAR_alpaca_secret
}

# ===== INSTANCE CONFIGURATION =====

variable "instance_type" {
  description = "EC2 instance type"
  type        = string
  default     = "t4g.small"

  validation {
    condition     = can(regex("^t4g\\.", var.instance_type))
    error_message = "Instance type must be ARM64 (t4g family) for cost optimization."
  }
}

variable "root_volume_size" {
  description = "Size of root EBS volume in GB"
  type        = number
  default     = 8

  validation {
    condition     = var.root_volume_size >= 8 && var.root_volume_size <= 100
    error_message = "Root volume size must be between 8 and 100 GB."
  }
}

variable "delete_volume_on_termination" {
  description = "Delete EBS volume when instance is terminated (set to false for production)"
  type        = bool
  default     = false  # Protect data by default
}

variable "environment" {
  description = "Environment name (dev, staging, production)"
  type        = string
  default     = "production"

  validation {
    condition     = contains(["dev", "staging", "production"], var.environment)
    error_message = "Environment must be dev, staging, or production."
  }
}

# ===== OPTIONAL FEATURES =====

variable "create_elastic_ip" {
  description = "Create Elastic IP for static IP address (costs $3.60/month if instance is stopped)"
  type        = bool
  default     = false
}

variable "enable_detailed_monitoring" {
  description = "Enable detailed CloudWatch monitoring (costs $2.10/month extra)"
  type        = bool
  default     = false
}

variable "create_cloudwatch_logs" {
  description = "Create CloudWatch log group for bot logs"
  type        = bool
  default     = false
}

variable "log_retention_days" {
  description = "CloudWatch log retention in days (only if create_cloudwatch_logs = true)"
  type        = number
  default     = 90  # Keep 3 months of trading bot logs

  validation {
    condition     = contains([1, 3, 5, 7, 14, 30, 60, 90, 120, 150, 180, 365, 400, 545, 731, 1827, 3653], var.log_retention_days)
    error_message = "Log retention must be a valid CloudWatch retention period."
  }
}

variable "create_sns_alerts" {
  description = "Create SNS topic for email alerts"
  type        = bool
  default     = false
}

variable "alert_email" {
  description = "Email address for SNS alerts (only if create_sns_alerts = true)"
  type        = string
  default     = ""

  validation {
    condition     = var.alert_email == "" || can(regex("^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$", var.alert_email))
    error_message = "Alert email must be a valid email address or empty."
  }
}

variable "create_cloudwatch_alarms" {
  description = "Create CloudWatch alarms for instance monitoring"
  type        = bool
  default     = false
}

variable "enable_scheduled_start_stop" {
  description = "Enable automated start/stop of EC2 instance during market hours (saves ~70% on EC2 costs). Requires Elastic IP for static IP address when stopped. Starts at 9:00 AM ET, stops at 4:30 PM ET on weekdays."
  type        = bool
  default     = false
}

# ===== DISCORD BOT (OPTIONAL ADDON) =====

variable "discord_token" {
  description = "Discord bot token for the observability bot (optional - leave empty to disable)"
  type        = string
  sensitive   = true
  default     = ""
  # Recommended: Set via environment variable TF_VAR_discord_token
}

variable "anthropic_api_key" {
  description = "Anthropic API key for Claude-powered investigations (optional - required if discord_token is set)"
  type        = string
  sensitive   = true
  default     = ""
  # Recommended: Set via environment variable TF_VAR_anthropic_api_key
}

variable "discord_allowed_channels" {
  description = "Comma-separated Discord channel IDs allowed to use the bot (optional - leave empty for all channels)"
  type        = string
  default     = ""
}
