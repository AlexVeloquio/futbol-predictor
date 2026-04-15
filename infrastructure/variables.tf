variable "aws_region" {
  description = "Región de AWS"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Nombre del proyecto (usado como prefijo en recursos)"
  type        = string
  default     = "futbol-predictor"
}

variable "environment" {
  description = "Entorno de despliegue"
  type        = string
  default     = "dev"
}
