############################################################
# Predictor de Resultados de Fútbol — Infraestructura AWS
############################################################

terraform {
  required_version = ">= 1.5"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # Descomentar para usar backend remoto en S3
  # backend "s3" {
  #   bucket = "futbol-predictor-tfstate"
  #   key    = "terraform.tfstate"
  #   region = "us-east-1"
  # }
}

provider "aws" {
  region = var.aws_region
}

# ---- S3 Bucket (Data Lake + Model Artifacts) ----
resource "aws_s3_bucket" "data_lake" {
  bucket = "${var.project_name}-data-${var.environment}"
}

resource "aws_s3_bucket_versioning" "data_lake" {
  bucket = aws_s3_bucket.data_lake.id
  versioning_configuration {
    status = "Enabled"
  }
}

# ---- ECR Repositories ----
resource "aws_ecr_repository" "training" {
  name                 = "${var.project_name}-training"
  image_tag_mutability = "MUTABLE"
  force_delete         = true
}

resource "aws_ecr_repository" "inference" {
  name                 = "${var.project_name}-inference"
  image_tag_mutability = "MUTABLE"
  force_delete         = true
}

# ---- DynamoDB Table ----
resource "aws_dynamodb_table" "predictions" {
  name         = "${var.project_name}-predictions"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "match_key"

  attribute {
    name = "match_key"
    type = "S"
  }
}

# ---- IAM Role for Lambda ----
resource "aws_iam_role" "lambda_role" {
  name = "${var.project_name}-lambda-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "lambda.amazonaws.com"
      }
    }]
  })
}

resource "aws_iam_role_policy" "lambda_policy" {
  name = "${var.project_name}-lambda-policy"
  role = aws_iam_role.lambda_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = ["s3:GetObject", "s3:PutObject", "s3:ListBucket"]
        Resource = [
          aws_s3_bucket.data_lake.arn,
          "${aws_s3_bucket.data_lake.arn}/*"
        ]
      },
      {
        Effect   = "Allow"
        Action   = ["dynamodb:PutItem", "dynamodb:GetItem", "dynamodb:Query"]
        Resource = aws_dynamodb_table.predictions.arn
      },
      {
        Effect   = "Allow"
        Action   = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"]
        Resource = "arn:aws:logs:*:*:*"
      }
    ]
  })
}

# ---- Lambda Functions ----
resource "aws_lambda_function" "training" {
  function_name = "${var.project_name}-training"
  role          = aws_iam_role.lambda_role.arn
  package_type  = "Image"
  image_uri     = "${aws_ecr_repository.training.repository_url}:latest"
  timeout       = 300
  memory_size   = 1024

  environment {
    variables = {
      MODEL_BUCKET = aws_s3_bucket.data_lake.id
      MODEL_KEY    = "models/model.pkl"
      META_KEY     = "models/metadata.json"
    }
  }
}

resource "aws_lambda_function" "inference" {
  function_name = "${var.project_name}-inference"
  role          = aws_iam_role.lambda_role.arn
  package_type  = "Image"
  image_uri     = "${aws_ecr_repository.inference.repository_url}:latest"
  timeout       = 30
  memory_size   = 512

  environment {
    variables = {
      MODEL_BUCKET = aws_s3_bucket.data_lake.id
      MODEL_KEY    = "models/model.pkl"
      META_KEY     = "models/metadata.json"
      DYNAMO_TABLE = aws_dynamodb_table.predictions.name
    }
  }
}

# ---- API Gateway ----
resource "aws_apigatewayv2_api" "api" {
  name          = "${var.project_name}-api"
  protocol_type = "HTTP"
}

resource "aws_apigatewayv2_integration" "predict" {
  api_id             = aws_apigatewayv2_api.api.id
  integration_type   = "AWS_PROXY"
  integration_uri    = aws_lambda_function.inference.invoke_arn
  payload_format_version = "2.0"
}

resource "aws_apigatewayv2_route" "predict" {
  api_id    = aws_apigatewayv2_api.api.id
  route_key = "POST /predict"
  target    = "integrations/${aws_apigatewayv2_integration.predict.id}"
}

resource "aws_apigatewayv2_stage" "default" {
  api_id      = aws_apigatewayv2_api.api.id
  name        = "$default"
  auto_deploy = true
}

resource "aws_lambda_permission" "api_gw" {
  statement_id  = "AllowAPIGateway"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.inference.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.api.execution_arn}/*/*"
}

# ---- S3 Event → Lambda Training ----
resource "aws_s3_bucket_notification" "new_data" {
  bucket = aws_s3_bucket.data_lake.id

  lambda_function {
    lambda_function_arn = aws_lambda_function.training.arn
    events              = ["s3:ObjectCreated:*"]
    filter_prefix       = "data/"
    filter_suffix       = ".csv"
  }
}

resource "aws_lambda_permission" "s3_trigger" {
  statement_id  = "AllowS3Invoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.training.function_name
  principal     = "s3.amazonaws.com"
  source_arn    = aws_s3_bucket.data_lake.arn
}
