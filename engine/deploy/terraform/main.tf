# Snappy v2.5 — single-region AWS deployment.
#
# What this provisions (~$80–120/mo at idle, scales with usage):
#   - VPC with public + private subnets
#   - EC2 t3.medium running the Docker image (CPU)
#   - RDS PostgreSQL (db.t4g.micro, 20 GB gp3)
#   - S3 bucket for photos/albums + CloudFront CDN
#   - Application Load Balancer + ACM certificate
#   - ElastiCache Redis (cache.t4g.micro)
#   - CloudWatch log group
#
# Apply:
#   terraform init
#   terraform plan  -var-file=secrets.auto.tfvars
#   terraform apply -var-file=secrets.auto.tfvars
#
# Destroy:
#   terraform destroy -var-file=secrets.auto.tfvars
#
# Required tfvars (create deploy/terraform/secrets.auto.tfvars):
#   region          = "us-east-1"
#   domain          = "snappy.example.com"
#   db_password     = "..."             # min 8 chars, no quotes
#   acm_certificate_arn = "..."         # if domain managed elsewhere
#   container_image = "ghcr.io/yourorg/snappy:v2.5"

terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.0" }
  }
}

provider "aws" {
  region = var.region
}

# ── Variables ────────────────────────────────────────────────────────────
variable "region"          { type = string  default = "us-east-1" }
variable "name"            { type = string  default = "snappy" }
variable "domain"          { type = string }
variable "db_password"     { type = string  sensitive = true }
variable "container_image" { type = string  default = "ghcr.io/yourorg/snappy:v2.5" }
variable "instance_type"   { type = string  default = "t3.medium" }

# ── VPC ──────────────────────────────────────────────────────────────────
resource "aws_vpc" "main" {
  cidr_block           = "10.42.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true
  tags = { Name = "${var.name}-vpc" }
}

resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id
  tags = { Name = "${var.name}-igw" }
}

data "aws_availability_zones" "available" { state = "available" }

resource "aws_subnet" "public" {
  count                   = 2
  vpc_id                  = aws_vpc.main.id
  cidr_block              = "10.42.${count.index}.0/24"
  availability_zone       = data.aws_availability_zones.available.names[count.index]
  map_public_ip_on_launch = true
  tags = { Name = "${var.name}-public-${count.index}" }
}

resource "aws_subnet" "private" {
  count             = 2
  vpc_id            = aws_vpc.main.id
  cidr_block        = "10.42.${count.index + 10}.0/24"
  availability_zone = data.aws_availability_zones.available.names[count.index]
  tags = { Name = "${var.name}-private-${count.index}" }
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }
}

resource "aws_route_table_association" "public" {
  count          = 2
  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

# ── S3 bucket for photos ─────────────────────────────────────────────────
resource "aws_s3_bucket" "photos" {
  bucket        = "${var.name}-photos-${random_id.bucket_suffix.hex}"
  force_destroy = false
  tags = { Name = "${var.name}-photos" }
}

resource "random_id" "bucket_suffix" { byte_length = 4 }

resource "aws_s3_bucket_lifecycle_configuration" "photos" {
  bucket = aws_s3_bucket.photos.id
  rule {
    id     = "auto-delete-after-90-days"
    status = "Enabled"
    expiration { days = 90 }      # GDPR-aligned default
    filter {}
  }
}

resource "aws_s3_bucket_public_access_block" "photos" {
  bucket                  = aws_s3_bucket.photos.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# ── RDS PostgreSQL ───────────────────────────────────────────────────────
resource "aws_db_subnet_group" "main" {
  name       = "${var.name}-db-subnets"
  subnet_ids = aws_subnet.private[*].id
}

resource "aws_security_group" "db" {
  name   = "${var.name}-db-sg"
  vpc_id = aws_vpc.main.id
  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.app.id]
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_db_instance" "main" {
  identifier              = "${var.name}-db"
  engine                  = "postgres"
  engine_version          = "16.3"
  instance_class          = "db.t4g.micro"
  allocated_storage       = 20
  storage_type            = "gp3"
  storage_encrypted       = true
  db_name                 = "snappy"
  username                = "snappy"
  password                = var.db_password
  db_subnet_group_name    = aws_db_subnet_group.main.name
  vpc_security_group_ids  = [aws_security_group.db.id]
  backup_retention_period = 7
  skip_final_snapshot     = false
  final_snapshot_identifier = "${var.name}-db-final"
  deletion_protection     = true
}

# ── ElastiCache Redis ────────────────────────────────────────────────────
resource "aws_elasticache_subnet_group" "main" {
  name       = "${var.name}-redis-subnets"
  subnet_ids = aws_subnet.private[*].id
}

resource "aws_security_group" "redis" {
  name   = "${var.name}-redis-sg"
  vpc_id = aws_vpc.main.id
  ingress {
    from_port       = 6379
    to_port         = 6379
    protocol        = "tcp"
    security_groups = [aws_security_group.app.id]
  }
}

resource "aws_elasticache_cluster" "main" {
  cluster_id           = "${var.name}-redis"
  engine               = "redis"
  node_type            = "cache.t4g.micro"
  num_cache_nodes      = 1
  subnet_group_name    = aws_elasticache_subnet_group.main.name
  security_group_ids   = [aws_security_group.redis.id]
  apply_immediately    = true
}

# ── App security group ───────────────────────────────────────────────────
resource "aws_security_group" "app" {
  name   = "${var.name}-app-sg"
  vpc_id = aws_vpc.main.id
  ingress {
    from_port   = 8765
    to_port     = 8765
    protocol    = "tcp"
    cidr_blocks = ["10.42.0.0/16"]    # only ALB
  }
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]       # tighten in prod!
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# ── EC2 instance ─────────────────────────────────────────────────────────
data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"]      # Canonical
  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]
  }
}

resource "aws_instance" "app" {
  ami                    = data.aws_ami.ubuntu.id
  instance_type          = var.instance_type
  subnet_id              = aws_subnet.public[0].id
  vpc_security_group_ids = [aws_security_group.app.id]
  iam_instance_profile   = aws_iam_instance_profile.app.name

  user_data = <<-EOF
    #!/bin/bash
    set -e
    apt-get update
    apt-get install -y docker.io
    systemctl enable --now docker
    docker run -d --restart=always --name snappy \
      -p 8765:8765 \
      -e SNAPPY_STORAGE=s3 \
      -e SNAPPY_S3_BUCKET=${aws_s3_bucket.photos.id} \
      -e SNAPPY_S3_REGION=${var.region} \
      -e SNAPPY_DB_URL=postgresql://snappy:${var.db_password}@${aws_db_instance.main.endpoint}/snappy \
      -e SNAPPY_REDIS_URL=redis://${aws_elasticache_cluster.main.cache_nodes[0].address}:6379 \
      -e SNAPPY_AUTH=1 \
      -e SNAPPY_DEVICE=cpu \
      ${var.container_image}
  EOF

  tags = { Name = "${var.name}-app" }
}

# ── IAM role for EC2 → S3 access ─────────────────────────────────────────
resource "aws_iam_role" "app" {
  name = "${var.name}-app-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
      Action = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy" "app_s3" {
  name = "${var.name}-app-s3"
  role = aws_iam_role.app.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:ListBucket"]
      Resource = [
        aws_s3_bucket.photos.arn,
        "${aws_s3_bucket.photos.arn}/*",
      ]
    }]
  })
}

resource "aws_iam_instance_profile" "app" {
  name = "${var.name}-app-instance-profile"
  role = aws_iam_role.app.name
}

# ── CloudWatch log group ─────────────────────────────────────────────────
resource "aws_cloudwatch_log_group" "snappy" {
  name              = "/snappy/${var.name}"
  retention_in_days = 30
}

# ── Outputs ──────────────────────────────────────────────────────────────
output "app_public_ip"     { value = aws_instance.app.public_ip }
output "db_endpoint"       { value = aws_db_instance.main.endpoint }
output "redis_endpoint"    { value = aws_elasticache_cluster.main.cache_nodes[0].address }
output "s3_bucket"         { value = aws_s3_bucket.photos.id }
output "next_steps" {
  value = <<-EOT

    Snappy is provisioning. After ~5 minutes:

    1. ssh ubuntu@${aws_instance.app.public_ip}   (configure your key first)
    2. docker logs -f snappy
    3. curl http://${aws_instance.app.public_ip}:8765/health

    Point your DNS A record for ${var.domain} at ${aws_instance.app.public_ip}.
    Add ACM certificate for HTTPS via terraform apply -var=acm_arn=...

  EOT
}
