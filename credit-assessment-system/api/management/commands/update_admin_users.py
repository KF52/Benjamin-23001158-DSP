from django.core.management.base import BaseCommand
from api.models import User, Role

class Command(BaseCommand):
    help = 'Update all admin users to have is_staff=True to avoid conflict with Django admin URLs'

    def handle(self, *args, **options):
        # Get all users with ADMIN role
        admin_users = User.objects.filter(role=Role.ADMIN)
        admin_count = admin_users.count()
        
        # Update admin users to have is_staff=True
        updated_count = 0
        for user in admin_users:
            if not user.is_staff:
                user.is_staff = True
                user.save()
                updated_count += 1
                self.stdout.write(self.style.SUCCESS(f'Updated user {user.email} - is_staff set to True'))
        
        # Summary message
        self.stdout.write(self.style.SUCCESS(
            f'Processed {admin_count} admin users. Updated {updated_count} users to have is_staff=True.'
        ))