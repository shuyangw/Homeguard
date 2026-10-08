# Homeguard Console Phase 2a: snapshot bucket, the instance's upload grant,
# and a read-only IAM user for the local console app.
# Access keys for the user are created by the operator with the CLI so they
# never enter Terraform state.

data "aws_caller_identity" "current" {}

resource "aws_s3_bucket" "console_snapshots" {
  bucket = var.console_snapshot_bucket

  tags = {
    Name = "homeguard-console-snapshots"
  }
}

resource "aws_s3_bucket_public_access_block" "console_snapshots" {
  bucket                  = aws_s3_bucket.console_snapshots.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "console_snapshots" {
  bucket = aws_s3_bucket.console_snapshots.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_iam_role_policy" "console_snapshot_upload" {
  count = var.enable_cloudwatch_agent ? 1 : 0

  name = "homeguard-console-snapshot-upload"
  role = aws_iam_role.ec2_cloudwatch[0].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = "s3:PutObject"
      Resource = "${aws_s3_bucket.console_snapshots.arn}/console/latest/*"
    }]
  })
}

resource "aws_iam_user" "console" {
  name = "homeguard-console"

  tags = {
    Name = "homeguard-console"
  }
}

resource "aws_iam_user_policy" "console_read" {
  name = "homeguard-console-read"
  user = aws_iam_user.console.name

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid      = "DescribeInstance"
        Effect   = "Allow"
        Action   = "ec2:DescribeInstances"
        Resource = "*"
      },
      {
        Sid      = "ReadSchedules"
        Effect   = "Allow"
        Action   = "scheduler:GetSchedule"
        Resource = "*"
      },
      {
        Sid      = "ReadSnapshot"
        Effect   = "Allow"
        Action   = "s3:GetObject"
        Resource = "${aws_s3_bucket.console_snapshots.arn}/console/latest/*"
      },
      {
        # Without ListBucket a missing object reads as AccessDenied; the bucket holds only the snapshot.
        Sid      = "ListSnapshotPrefix"
        Effect   = "Allow"
        Action   = "s3:ListBucket"
        Resource = aws_s3_bucket.console_snapshots.arn
      },
      {
        Sid    = "ReadSchedulerLambdaLogs"
        Effect = "Allow"
        Action = "logs:FilterLogEvents"
        Resource = [
          "arn:aws:logs:${var.aws_region}:${data.aws_caller_identity.current.account_id}:log-group:/aws/lambda/homeguard-start-instance:*",
          "arn:aws:logs:${var.aws_region}:${data.aws_caller_identity.current.account_id}:log-group:/aws/lambda/homeguard-stop-instance:*",
        ]
      },
    ]
  })
}

output "console_snapshot_bucket" {
  value = aws_s3_bucket.console_snapshots.bucket
}
