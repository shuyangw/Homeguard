#!/bin/bash
# Quick helper to get your current public IP for AWS security groups

echo "========================================"
echo "Your Current Public IP Address"
echo "========================================"
echo ""

IP=$(curl -s https://checkip.amazonaws.com)

echo "IP Address: $IP"
echo "For AWS:    $IP/32"
echo ""
echo "Use this IP in AWS Security Group rules:"
echo "  Type: SSH"
echo "  Port: 22"
echo "  Source: $IP/32"
echo ""
echo "Security Group: homeguard-trading-bot-sg"
echo "Region: us-east-1"
echo ""
echo "Direct link:"
echo "https://us-east-1.console.aws.amazon.com/ec2/home?region=us-east-1#SecurityGroups:search=homeguard-trading-bot-sg"
echo ""
