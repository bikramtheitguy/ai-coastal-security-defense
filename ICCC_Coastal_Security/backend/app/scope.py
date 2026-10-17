"""Shared object policy for the synthetic POC. Capabilities are checked by routes.

Station/district command is scoped to current posting. Field recipients retain
access to explicitly tasked incidents across station boundaries. Unassigned
intake is visible to state triage only. Null/unknown postings never widen access.
"""
from fastapi import HTTPException
from sqlalchemy import and_, or_, true

from .audit import audit
from .models import Alert, Asset, Conversation, Incident, Order, Personnel, Station, Vessel
from .rbac import can_act_on_station, station_scope

INTEL_ALERTS = {"WATCHLIST", "IDENTITY_MISMATCH", "REPEATED_VISITS"}


def scope_station_query(db, user, query, column):
    scope = station_scope(db, user)
    return query if scope is None else query.filter(column.in_(scope))


def alert_query(db, user):
    q = scope_station_query(db, user, db.query(Alert), Alert.station_id)
    if "INTEL_VIEW" not in user._perms:
        q = q.filter(Alert.alert_type.notin_(INTEL_ALERTS))
    return q


def incident_query(db, user):
    from .services.orders import is_recipient
    scope = station_scope(db, user)
    q = db.query(Incident)
    if scope is not None:
        # An authenticated field assignment is explicit delegation to the incident.
        delegated = [o.incident_id for o in db.query(Order).filter(Order.incident_id.isnot(None))
                     if "ORDERS_FIELD" in user._perms and is_recipient(db, user, o)]
        condition = Incident.station_id.in_(scope)
        if user.jurisdiction == "UNIT":
            condition = Incident.id.in_(delegated)
        else:
            condition = or_(condition, Incident.id.in_(delegated))
        q = q.filter(condition)
    if "INTEL_VIEW" not in user._perms:
        q = q.filter(or_(Incident.alert_id.is_(None), Incident.alert_id.notin_(
            db.query(Alert.id).filter(Alert.alert_type.in_(INTEL_ALERTS)))))
    return q


def conversation_query(db, user):
    q = db.query(Conversation)
    scope = station_scope(db, user)
    linked = Conversation.incident_id.in_(incident_query(db, user).with_entities(Incident.id))
    unlinked = true() if scope is None else Conversation.vessel_id.in_(
        db.query(Vessel.id).filter(Vessel.station_id.in_(scope)))
    return q.filter(or_(linked, and_(Conversation.incident_id.is_(None), unlinked)))


def deny(db, user, kind, object_id):
    # Call before any mutation so denial commits cannot persist partial writes.
    audit(db, user=user, action="ACCESS_DENIED", entity_type=kind, entity_id=object_id,
          outcome="DENIED", detail="Object outside current jurisdiction or need-to-know")
    db.commit()
    raise HTTPException(403, "Outside your jurisdiction or authorised object scope")


def ensure_incident(db, user, inc):
    if inc is None:
        raise HTTPException(404, "Incident not found")
    if not incident_query(db, user).filter(Incident.id == inc.id).first():
        deny(db, user, "incident", inc.id)


def ensure_alert(db, user, alert):
    if alert is None:
        raise HTTPException(404, "Alert not found")
    if not alert_query(db, user).filter(Alert.id == alert.id).first():
        deny(db, user, "alert", alert.id)


def ensure_conversation(db, user, conversation):
    if conversation is None:
        raise HTTPException(404, "Conversation not found")
    if not conversation_query(db, user).filter(Conversation.id == conversation.id).first():
        deny(db, user, "conversation", conversation.id)


def ensure_evidence(db, user, evidence):
    if evidence is None:
        raise HTTPException(404, "Evidence not found")
    ensure_incident(db, user, db.get(Incident, evidence.incident_id) if evidence.incident_id else None)


def order_in_scope(db, user, order):
    """Command delegation requires jurisdiction over EVERY referenced target.

    Issuer identity alone never bypasses a changed posting. Recipient delegation
    permits field execution/read, not cancellation of an out-of-scope order.
    """
    stations = []
    for model, key in ((Asset, order.asset_id), (Incident, order.incident_id)):
        if key is not None:
            obj = db.get(model, key)
            if obj is None:
                return False
            stations.append(obj.station_id)
    for target in order.recipients or []:
        model = {"STATION": Station, "ASSET": Asset, "PERSONNEL": Personnel}.get(target.get("kind"))
        key = target.get("id")
        if model is None or type(key) is not int or key <= 0:
            return False
        obj = db.get(model, key)
        if obj is None:
            return False
        stations.append(obj.id if model is Station else obj.station_id)
    if not stations:
        return order.issuer == user.username and station_scope(db, user) != set()
    return all(can_act_on_station(db, user, sid) for sid in stations)


def can_read_order(db, user, order):
    from .services.orders import is_recipient
    if order.incident_id and not incident_query(db, user).filter(Incident.id == order.incident_id).first():
        return False
    if "ORDERS_FIELD" in user._perms and is_recipient(db, user, order):
        return True
    return order_in_scope(db, user, order)


def ensure_order(db, user, order):
    if not can_read_order(db, user, order):
        deny(db, user, "order", order.id)


def validate_order_targets(db, user, body):
    if body.incident_id is not None:
        inc = db.get(Incident, body.incident_id)
        ensure_incident(db, user, inc)
        if not can_act_on_station(db, user, inc.station_id):
            deny(db, user, "incident", inc.id)
    if body.asset_id is not None:
        asset = db.get(Asset, body.asset_id)
        if asset is None:
            raise HTTPException(404, "Asset not found")
        if not can_act_on_station(db, user, asset.station_id):
            deny(db, user, "asset", asset.id)
    for target in body.recipients:
        model = {"STATION": Station, "ASSET": Asset, "PERSONNEL": Personnel}.get(target.get("kind"))
        key = target.get("id")
        if model is None or type(key) is not int or key <= 0:
            raise HTTPException(400, "Recipient needs a valid kind and positive integer id")
        obj = db.get(model, key)
        if obj is None:
            raise HTTPException(400, "Recipient does not exist")
        sid = obj.id if model is Station else obj.station_id
        if not can_act_on_station(db, user, sid):
            deny(db, user, "recipient", key)


def feed_visible(db, user, entry):
    if entry.ref_type == "incident":
        return "INCIDENT_VIEW" in user._perms and bool(incident_query(db, user).filter(Incident.id == entry.ref_id).first())
    if entry.ref_type == "alert":
        return "INCIDENT_VIEW" in user._perms and bool(alert_query(db, user).filter(Alert.id == entry.ref_id).first())
    if entry.ref_type == "conversation":
        return "CHAT_VIEW" in user._perms and bool(conversation_query(db, user).filter(Conversation.id == entry.ref_id).first())
    if entry.ref_type == "order":
        order = db.get(Order, entry.ref_id)
        return order is not None and can_read_order(db, user, order)
    return can_act_on_station(db, user, entry.station_id)
