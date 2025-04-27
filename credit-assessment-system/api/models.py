from django.db import models
from django.contrib.auth.models import AbstractBaseUser, BaseUserManager, PermissionsMixin
from django.conf import settings
from django import forms

# User Manager
class UserManager(BaseUserManager):
    def create_user(self, email, name, password=None, role='End User'):
        ## Email cannot be an empty field as it is an unique identifier
        if not email:
            raise ValueError("Users must have an email address")
        email = self.normalize_email(email)
        user = self.model(email=email, name=name, role=role)
        user.set_password(password)
        user.save(using=self._db)
        return user
    
    def create_superuser(self, email, name, password=None):
        user = self.create_user(email, name, password, role='Admin')
        user.is_staff = True
        user.is_superuser = True
        user.save(using=self._db)
        return user
    
# Role Choices 
class Role(models.TextChoices):
    END_USER = "End User", "End User" # Loan Applicants
    ADMIN = "Admin", "Admin" # System Administrator
    AI_ENGINEER = "AI Engineer", "AI Engineer" # ML/AI Engineers
    STAFF = "Staff", "Staff" # Bank or Financial Institution Officers

# User Model (All Users)
class User(AbstractBaseUser, PermissionsMixin):
    email = models.EmailField(unique=True)
    name = models.CharField(max_length=255)
    role = models.CharField(
        max_length=50,
        choices=Role.choices,
        default=Role.END_USER
    )
    is_staff = models.BooleanField(default=False)
    member_since = models.DateTimeField(auto_now_add=True)
    
    objects = UserManager()

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['name']
    
    def save(self, *args, **kwargs):
        # First user gets Admin role
        if not User.objects.exists():  
            self.role = Role.ADMIN
        
        # Set is_staff=True for Admin users
        if self.role == Role.ADMIN:
            self.is_staff = True
            
        super().save(*args, **kwargs)

    def __str__(self):
        return self.email
    
# Replace Claim model with CreditApplication
class LoanApplication(models.Model):
    
    ##### fields pending #####

    def __str__(self):
        return f"Application {self.client_id} - {self.user.username}"

# Prediction Results
class Prediction(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='predictions')
    timestamp = models.DateTimeField(auto_now_add=True)
    input_data = models.JSONField()
    result = models.JSONField()
    loan_approval = models.BooleanField(default=False)
    confidence_score = models.DecimalField(max_digits=10, decimal_places=5, null=True, blank=True)    
    is_satisfactory = models.BooleanField(null=True, default=None)
    applicant_comments = models.TextField(null=True, blank=True)
    needs_review = models.BooleanField(default=False)
    feedback_date = models.DateTimeField(null=True, blank=True)
    is_checked = models.BooleanField(default=False)

    modified_by_staff = models.BooleanField(default=False)
    staff_comments = models.TextField(null=True, blank=True, verbose_name="Staff Comments")
    modified_by = models.ForeignKey(settings.AUTH_USER_MODEL, 
                                    null=True, blank=True, 
                                    on_delete=models.SET_NULL,
                                    related_name="modified_predictions")
    modified_at = models.DateTimeField(null=True, blank=True)
    original_prediction = models.BooleanField(null=True, blank=True)
    
    class Meta:
        ordering = ['-timestamp']
    
    def __str__(self):
        return f"Prediction for {self.user.name}: {'Approved' if self.loan_approval else 'Rejected'}" 
    
class PredictionUpdateForm(forms.ModelForm):
    class Meta:
        model = Prediction
        fields = ['loan_approval', 'staff_comments']
        widgets = {
            'staff_comments': forms.Textarea(attrs={'rows': 3}),
        }
        labels = {
            'loan_approval': 'Loan Status',
            'staff_comments': 'Comments for Applicant'
        }
    
## Machine Learning Model Data 
class MLModel(models.Model):
    model_name = models.CharField(max_length=100)
    file = models.FileField(upload_to='ml_models/')
    model_type = models.CharField(max_length=50)
    description = models.TextField(blank=True)
    uploaded_by = models.ForeignKey(User, on_delete=models.CASCADE)
    upload_date = models.DateTimeField(auto_now_add=True)
    is_active = models.BooleanField(default=False)
    requires_scaling = models.BooleanField(default=False)
    
    def __str__(self):
        status = "Active" if self.is_active else "Inactive"
        return f"{self.model_name} ({self.model_type}) - {status}"
    
    def save(self, *args, **kwargs):
        # If this model is being set as active, deactivate ALL other models
        if self.is_active:
            MLModel.objects.filter(is_active=True).exclude(id=self.id).update(is_active=False)
        super().save(*args, **kwargs)

# API Metrics Model for System Health Tracking
class APIMetrics(models.Model):
    """Model for tracking API performance metrics"""
    endpoint = models.CharField(max_length=255, help_text="API endpoint path")
    response_time = models.FloatField(help_text="Response time in milliseconds")
    status_code = models.IntegerField(help_text="HTTP status code")
    error = models.BooleanField(default=False, help_text="Whether this request resulted in an error")
    timestamp = models.DateTimeField(auto_now_add=True, help_text="When this metric was recorded")
    
    def __str__(self):
        return f"{self.endpoint} - {self.status_code} - {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
    
    class Meta:
        verbose_name = "API Metric"
        verbose_name_plural = "API Metrics"
        ordering = ["-timestamp"]