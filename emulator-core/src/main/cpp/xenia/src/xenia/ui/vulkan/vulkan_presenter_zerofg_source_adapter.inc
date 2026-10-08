/**
 ******************************************************************************
 * XenDroid ZeroFG Source handoff adapter                                    *
 ******************************************************************************
 */

// Source-side adapter for the ZeroFG independent presenter. The normal
// XenDroid path remains authoritative whenever ZeroFG is unavailable.

bool VulkanPresenter::AcquireZeroFGIngressSource(
    ZeroFGIndependentPresenter::IngressSourcePublication& publication_out) {
  publication_out = {};
  if (zerofg_vulkan_context_->handoff) {
    if (zerofg_vulkan_context_->handoff->failed() || ZeroFGDevice()->IsLost()) {
      publication_out.external_failure = true;
      return true;
    }
    const bool acquired =
        zerofg_vulkan_context_->handoff->Acquire(publication_out);
    // Acquire marks a bounded reporting window while holding the pool lock;
    // emit it only after the lock has been released on the presenter thread.
    zerofg_vulkan_context_->handoff->MaybeLog();
    return acquired;
  }
  uint32_t mailbox_index = UINT32_MAX;
  GuestOutputProperties properties;
  std::unique_lock<std::mutex> consumer_lock = ConsumeGuestOutput(
      mailbox_index, &properties, nullptr, nullptr);
  if (mailbox_index == UINT32_MAX || mailbox_index >= guest_output_images_.size()) {
    return false;
  }
  const std::shared_ptr<GuestOutputImage>& image =
      guest_output_images_[mailbox_index].image;
  if (!image || !properties.source_publication.publication_id) {
    return false;
  }

  publication_out.mailbox_index = mailbox_index;
  publication_out.content_width = properties.frontbuffer_width;
  publication_out.content_height = properties.frontbuffer_height;
  publication_out.display_aspect_ratio_x = properties.display_aspect_ratio_x;
  publication_out.display_aspect_ratio_y = properties.display_aspect_ratio_y;
  publication_out.is_8bpc = properties.is_8bpc;
  publication_out.publication = properties.source_publication;
  publication_out.image = image->image();
  publication_out.view = image->view();
  publication_out.extent = image->extent();
  publication_out.layout = kGuestOutputInternalLayout;
  publication_out.image_lifetime = std::static_pointer_cast<void>(image);
  publication_out.mailbox_acquisition = std::shared_ptr<void>(
      new std::unique_lock<std::mutex>(std::move(consumer_lock)),
      [](void* lock) {
        delete static_cast<std::unique_lock<std::mutex>*>(lock);
      });
  return true;
}

bool VulkanPresenter::OnGuestOutputPublished(
    uint32_t mailbox_index, bool guest_output_active, uint64_t source_id,
    uint64_t issue_time_ns, uint64_t publish_time_ns,
    const GuestOutputProperties& properties) {
  if (!guest_output_active || !zerofg_independent_presenter_ ||
      !zerofg_independent_presenter_->accepting()) {
    return false;
  }
  if (zerofg_vulkan_context_->handoff) {
    ZeroFGIndependentPresenter::IngressSourcePublication metadata;
    metadata.mailbox_index = mailbox_index;
    metadata.content_width = properties.frontbuffer_width;
    metadata.content_height = properties.frontbuffer_height;
    metadata.display_aspect_ratio_x = properties.display_aspect_ratio_x;
    metadata.display_aspect_ratio_y = properties.display_aspect_ratio_y;
    metadata.is_8bpc = properties.is_8bpc;
    metadata.publication = properties.source_publication;
    zerofg_vulkan_context_->handoff->Publish(
        mailbox_index, metadata, zerofg_independent_presenter_.get());
    // Authority is not acceptance. Even a skipped publication wakes B to
    // prepare the requested extent, without forcing Source through native WSI.
  }
  zerofg_independent_presenter_->ClaimPublicationAuthority();
  return true;
}

void VulkanPresenter::OnGuestOutputPublicationCommitted() {
  if (zerofg_independent_presenter_) {
    zerofg_independent_presenter_->PublicationCommitted();
    return;
  }
}
