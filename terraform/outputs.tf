# Terraform Outputs - Connection and Resource Information

output "instance_id" {
  description = "EC2 instance ID"
  value       = aws_instance.trading_bot.id
}

output "instance_public_ip" {
  description = "Public IP address of the EC2 instance"
  value       = var.create_elastic_ip ? aws_eip.trading_bot[0].public_ip : aws_instance.trading_bot.public_ip
}

output "instance_public_dns" {
  description = "Public DNS name of the EC2 instance"
  value       = aws_instance.trading_bot.public_dns
}

output "elastic_ip" {
  description = "Elastic IP address (if created)"
  value       = var.create_elastic_ip ? aws_eip.trading_bot[0].public_ip : null
}

output "ssh_connection_command" {
  description = "SSH command to connect to the instance"
  value       = "ssh -i ~/.ssh/${var.key_pair_name}.pem ec2-user@${var.create_elastic_ip ? aws_eip.trading_bot[0].public_ip : aws_instance.trading_bot.public_ip}"
}

output "security_group_id" {
  description = "Security group ID"
  value       = aws_security_group.trading_bot.id
}

output "ami_id" {
  description = "AMI ID used for the instance"
  value       = aws_instance.trading_bot.ami
}

output "ami_name" {
  description = "AMI name (Amazon Linux 2023 ARM64)"
  value       = data.aws_ami.amazon_linux_2023_arm64.name
}

output "cloudwatch_log_group" {
  description = "CloudWatch log group name (if created)"
  value       = var.create_cloudwatch_logs ? aws_cloudwatch_log_group.trading_logs[0].name : null
}

output "sns_topic_arn" {
  description = "SNS topic ARN for alerts (if created)"
  value       = var.create_sns_alerts ? aws_sns_topic.trading_alerts[0].arn : null
}

output "instance_state" {
  description = "Current state of the EC2 instance"
  value       = aws_instance.trading_bot.instance_state
}

output "root_volume_size" {
  description = "Root volume size in GB"
  value       = var.root_volume_size
}

output "estimated_monthly_cost" {
  description = "Estimated monthly cost in USD (EC2 + EBS only, approximate)"
  value       = format("$%.2f", (var.instance_type == "t4g.small" ? 12.26 : var.instance_type == "t4g.medium" ? 24.53 : 0) + (var.root_volume_size * 0.08))
}

output "post_deployment_instructions" {
  description = "Next steps after deployment"
  value       = <<-EOT
    ==========================================
    Homeguard Trading Bot Deployed Successfully!
    ==========================================

    Instance ID: ${aws_instance.trading_bot.id}
    Public IP:   ${var.create_elastic_ip ? aws_eip.trading_bot[0].public_ip : aws_instance.trading_bot.public_ip}

    Next Steps:
    -----------
    1. Wait 3-5 minutes for user-data script to complete

    2. SSH to the instance:
       ${format("ssh -i ~/.ssh/%s.pem ec2-user@%s", var.key_pair_name, var.create_elastic_ip ? aws_eip.trading_bot[0].public_ip : aws_instance.trading_bot.public_ip)}

    3. Check installation progress:
       tail -f /var/log/cloud-init-output.log

    4. Verify bot is running:
       sudo systemctl status homeguard-trading

    5. View logs:
       tail -f ~/logs/trading_$(date +%%Y%%m%%d).log

    6. Monitor service:
       sudo journalctl -u homeguard-trading -f

    ==========================================
    EOT
}
