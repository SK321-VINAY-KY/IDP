# Antigravity Agent Guidelines

## Mandatory AWS Resource Tags
Whenever creating, configuring, or provisioning any AWS service or resource (including but not limited to ECS Task Definitions/Services/Tasks, Lambda functions, RDS databases, ECR repositories, S3 buckets, IAM roles/policies, and CloudWatch log groups), ALWAYS attach the following 3 mandatory tags:

- **createdby**: `vinay.k@shellkode`
- **purpose**: `internal`
- **customer**: `internal`

### Tag Formats Reference:
- **AWS CLI (ECS / Lambda / Generic)**:
  `--tags key=createdby,value=vinay.k@shellkode key=purpose,value=internal key=customer,value=internal`
- **AWS CLI (RDS / S3)**:
  `--tags Key=createdby,Value=vinay.k@shellkode Key=purpose,Value=internal Key=customer,Value=internal`
- **Boto3 API**:
  ```python
  tags = [
      {"Key": "createdby", "Value": "vinay.k@shellkode"},
      {"Key": "purpose", "Value": "internal"},
      {"Key": "customer", "Value": "internal"},
  ]
  ```
- **ECS Task Definition JSON**:
  ```json
  "tags": [
    {"key": "createdby", "value": "vinay.k@shellkode"},
    {"key": "purpose", "value": "internal"},
    {"key": "customer", "value": "internal"}
  ]
  ```
