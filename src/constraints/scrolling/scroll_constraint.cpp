#include "rive/constraints/scrolling/scroll_constraint.hpp"
#include "rive/constraints/scrolling/scroll_constraint_proxy.hpp"
#include "rive/constraints/transform_constraint.hpp"
#include "rive/core_context.hpp"
#include "rive/layout/layout_node_provider.hpp"
#include "rive/transform_component.hpp"
#include "rive/math/mat2d.hpp"

using namespace rive;

ScrollConstraint::~ScrollConstraint() { delete m_physics; }

void ScrollConstraint::constrain(TransformComponent* component)
{
    m_scrollTransform =
        Mat2D::fromTranslate(constrainsHorizontal() ? clampedOffsetX() : 0,
                             constrainsVertical() ? clampedOffsetY() : 0);
}

void ScrollConstraint::constrainChild(LayoutNodeProvider* child)
{
    auto component = child->transformComponent();
    if (component == nullptr)
    {
        return;
    }
    auto targetTransform =
        Mat2D::multiply(component->worldTransform(), m_scrollTransform);
    TransformConstraint::constrainWorld(component,
                                        component->worldTransform(),
                                        m_componentsA,
                                        targetTransform,
                                        m_componentsB,
                                        strength());
}

void ScrollConstraint::dragView(Vec2D delta)
{
    if (m_physics != nullptr)
    {
        m_physics->accumulate(delta);
    }
    scrollOffsetX(offsetX() + delta.x);
    scrollOffsetY(offsetY() + delta.y);
}

void ScrollConstraint::runPhysics()
{
    m_isDragging = false;
    std::vector<Vec2D> snappingPoints;
    if (snap())
    {
        for (auto child : content()->children())
        {
            auto c = LayoutNodeProvider::from(child);
            if (c != nullptr)
            {
                size_t count = c->numLayoutNodes();
                for (int j = 0; j < count; j++)
                {
                    auto bounds = c->layoutBoundsForNode(j);
                    snappingPoints.push_back(
                        Vec2D(bounds.left(), bounds.top()));
                }
            }
        }
    }
    if (m_physics != nullptr)
    {
        m_physics->run(Vec2D(maxOffsetX(), maxOffsetY()),
                       Vec2D(offsetX(), offsetY()),
                       snap() ? snappingPoints : std::vector<Vec2D>());
    }
}

bool ScrollConstraint::advanceComponent(float elapsedSeconds,
                                        AdvanceFlags flags)
{
    if ((flags & AdvanceFlags::AdvanceNested) != AdvanceFlags::AdvanceNested)
    {
        // offsetX(0);
        // offsetY(0);
        return false;
    }
    if (m_physics == nullptr)
    {
        return false;
    }
    if (m_physics->isRunning())
    {
        auto offset = m_physics->advance(elapsedSeconds);
        scrollOffsetX(offset.x);
        scrollOffsetY(offset.y);
    }
    return m_physics->enabled();
}

std::vector<DraggableProxy*> ScrollConstraint::draggables()
{
    std::vector<DraggableProxy*> items;
    items.push_back(new ViewportDraggableProxy(this, viewport()->proxy()));
    return items;
}

void ScrollConstraint::buildDependencies()
{
    Super::buildDependencies();
    for (auto child : content()->children())
    {
        auto layout = LayoutNodeProvider::from(child);
        if (layout != nullptr)
        {
            addDependent(child);
            layout->addLayoutConstraint(static_cast<LayoutConstraint*>(this));
        }
    }
}

Core* ScrollConstraint::clone() const
{
    auto cloned = ScrollConstraintBase::clone();
    if (physics() != nullptr)
    {
        auto constraint = cloned->as<ScrollConstraint>();
        auto clonedPhysics = physics()->clone()->as<ScrollPhysics>();
        constraint->physics(clonedPhysics);
    }
    return cloned;
}

StatusCode ScrollConstraint::import(ImportStack& importStack)
{
    auto backboardImporter =
        importStack.latest<BackboardImporter>(BackboardBase::typeKey);
    if (backboardImporter != nullptr)
    {
        std::vector<ScrollPhysics*> physicsObjects =
            backboardImporter->physics();
        if (physicsId() != -1 && physicsId() < physicsObjects.size())
        {
            auto phys = physicsObjects[physicsId()];
            if (phys != nullptr)
            {
                auto cloned = phys->clone()->as<ScrollPhysics>();
                physics(cloned);
            }
        }
    }
    else
    {
        return StatusCode::MissingObject;
    }
    return Super::import(importStack);
}

StatusCode ScrollConstraint::onAddedDirty(CoreContext* context)
{
    StatusCode result = Super::onAddedDirty(context);
    offsetX(scrollOffsetX());
    offsetY(scrollOffsetY());
    return result;
}

void ScrollConstraint::initPhysics()
{
    m_isDragging = true;
    if (m_physics != nullptr)
    {
        m_physics->prepare(direction());
    }
}

void ScrollConstraint::stopPhysics()
{
    if (m_physics != nullptr)
    {
        m_physics->reset();
    }
}

float ScrollConstraint::scrollPercentX()
{
    return maxOffsetX() != 0 ? scrollOffsetX() / maxOffsetX() : 0;
}

float ScrollConstraint::scrollPercentY()
{
    return maxOffsetY() != 0 ? scrollOffsetY() / maxOffsetY() : 0;
}

float ScrollConstraint::scrollIndex()
{
    return indexAtPosition(Vec2D(scrollOffsetX(), scrollOffsetY()));
}

void ScrollConstraint::setScrollPercentX(float value)
{
    if (m_isDragging)
    {
        return;
    }
    stopPhysics();
    float to = value * maxOffsetX();
    scrollOffsetX(to);
}

void ScrollConstraint::setScrollPercentY(float value)
{
    if (m_isDragging)
    {
        return;
    }
    stopPhysics();
    float to = value * maxOffsetY();
    scrollOffsetY(to);
}

void ScrollConstraint::setScrollIndex(float value)
{
    if (m_isDragging)
    {
        return;
    }
    stopPhysics();
    Vec2D to = positionAtIndex(value);
    if (constrainsHorizontal())
    {
        scrollOffsetX(to.x);
    }
    else if (constrainsVertical())
    {
        scrollOffsetY(to.y);
    }
}

Vec2D ScrollConstraint::positionAtIndex(float index)
{
    if (content() == nullptr || content()->children().size() == 0)
    {
        return Vec2D();
    }
    uint32_t i = 0;
    Vec2D contentGap = gap();
    float floorIndex = std::floor(index);
    LayoutNodeProvider* lastChild = nullptr;
    for (auto child : content()->children())
    {
        auto c = LayoutNodeProvider::from(child);
        if (c != nullptr)
        {
            size_t count = c->numLayoutNodes();
            if ((uint32_t)floorIndex < i + count)
            {
                float mod = index - floorIndex;
                auto bounds = c->layoutBoundsForNode(floorIndex - i);
                return Vec2D(
                    -bounds.left() - (bounds.width() + contentGap.x) * mod,
                    -bounds.top() - (bounds.height() + contentGap.y) * mod);
            }
            lastChild = c;
            i += count;
        }
    }
    if (lastChild == nullptr)
    {
        return Vec2D();
    }

    auto bounds =
        lastChild->layoutBoundsForNode((int)lastChild->numLayoutNodes() - 1);
    return Vec2D(-bounds.left(), -bounds.top());
}

float ScrollConstraint::indexAtPosition(Vec2D pos)
{
    if (content() == nullptr || content()->children().size() == 0)
    {
        return 0;
    }
    float i = 0.0f;
    Vec2D contentGap = gap();
    if (constrainsHorizontal())
    {
        for (auto child : content()->children())
        {
            auto c = LayoutNodeProvider::from(child);
            if (c != nullptr)
            {
                size_t count = c->numLayoutNodes();
                for (int j = 0; j < count; j++)
                {
                    auto bounds = c->layoutBoundsForNode(j);
                    if (pos.x >
                        -bounds.left() - (bounds.width() + contentGap.x))
                    {
                        return (i + j) + (-pos.x - bounds.left()) /
                                             (bounds.width() + contentGap.x);
                    }
                }
                i += count;
            }
        }
        return i;
    }
    else if (constrainsVertical())
    {
        for (auto child : content()->children())
        {
            auto c = LayoutNodeProvider::from(child);
            if (c != nullptr)
            {
                size_t count = c->numLayoutNodes();
                for (int j = 0; j < count; j++)
                {
                    auto bounds = c->layoutBoundsForNode(j);
                    if (pos.y >
                        -bounds.top() - (bounds.height() + contentGap.y))
                    {
                        return (i + j) + (-pos.y - bounds.top()) /
                                             (bounds.height() + contentGap.y);
                    }
                }
                i += count;
            }
        }
        return i;
    }
    return 0;
}

size_t ScrollConstraint::scrollItemCount()
{
    size_t count = 0;
    for (auto child : content()->children())
    {
        auto c = LayoutNodeProvider::from(child);
        if (c != nullptr)
        {
            count += c->numLayoutNodes();
        }
    }
    return count;
}

Vec2D ScrollConstraint::gap()
{
    if (content() == nullptr)
    {
        return Vec2D();
    }
    return Vec2D(content()->gapHorizontal(), content()->gapVertical());
}