from rest_framework import permissions 
from .models import Role

## AdminPermissions
class IsAdminUser(permissions.BasePermission):
    def has_permission(self, request, view):
        return request.user.role == Role.ADMIN

## FinanceTeamPermissions 
class IsStaff(permissions.BasePermission):
    def has_permission(self, request, view):
        return request.user.role == Role.STAFF
    
## AIEngineerPermissions 
class IsAIEngineer(permissions.BasePermission):
    def has_permission(self, request, view):
        return request.user.role == Role.AI_ENGINEER
    