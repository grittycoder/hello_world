// Models for the application, defining the structure of the database tables and the relationships between them.

from django.db import models

class Item(models.Model):
    name = models.CharField(max_length=100)
    description = models.TextField()
    price = models.DecimalField(max_digits=10, decimal_places=2)

    def __str__(self):
        return self.name
class Order(models.Model):
    item = models.ForeignKey(Item, on_delete=models.CASCADE)
    quantity = models.PositiveIntegerField()
    total_price = models.DecimalField(max_digits=10, decimal_places=2)

    def save(self, *args, **kwargs):
        self.total_price = self.item.price * self.quantity
        super().save(*args, **kwargs)

    def __str__(self):
        return f"Order of {self.quantity} {self.item.name}(s)"