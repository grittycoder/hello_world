// shared/pricingLogic.js
export const calculateTotal = (service, rooms, baths, addons = []) => {
  if (!service) return 0;
  
  const base = service.basePrice || 0;
  const roomCost = (rooms || 0) * (service.modifiers?.pricePerBedroom || 0);
  const bathCost = (baths || 0) * (service.modifiers?.pricePerBathroom || 0);
  const addonCost = addons.reduce((sum, addon) => sum + (addon.price || 0), 0);
  
  return base + roomCost + bathCost + addonCost;
};
