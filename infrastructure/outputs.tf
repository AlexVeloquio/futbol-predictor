output "api_endpoint" {
  description = "URL del API Gateway"
  value       = aws_apigatewayv2_api.api.api_endpoint
}

output "s3_bucket" {
  description = "Nombre del bucket S3"
  value       = aws_s3_bucket.data_lake.id
}

output "dynamodb_table" {
  description = "Nombre de la tabla DynamoDB"
  value       = aws_dynamodb_table.predictions.name
}

output "ecr_training_url" {
  description = "URL del repositorio ECR de training"
  value       = aws_ecr_repository.training.repository_url
}

output "ecr_inference_url" {
  description = "URL del repositorio ECR de inference"
  value       = aws_ecr_repository.inference.repository_url
}
