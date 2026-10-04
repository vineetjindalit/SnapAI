/* SnapAI shared site script: config, language, header/footer, tracking, API. */
window.SNAP = (function () {
  const CONFIG = {
    supabaseUrl: "https://jksqxvdoutwigirnqfea.supabase.co",
    anonKey: "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imprc3F4dmRvdXR3aWdpcm5xZmVhIiwicm9sZSI6ImFub24iLCJpYXQiOjE3OTA0MzU5MDUsImV4cCI6MjEwNjAxMTkwNX0.s3zAyS_OsdFnPjuWCg0mpLcpSC-QEY6C5QM7JWkCk8c",
    whatsapp: "918368492368",
    phoneDisplay: "+91 83684 92368",
    instagram: "https://www.instagram.com/",
    messenger: "https://m.me/1328785776984462",
    email: "",
    razorpayKeyId: "rzp_test_Tgl2BLcG6nRpRl",
    prices: { "30 min": 99, "1 hr": 199, "2 hr": 299, "3 hr": 399 },
    hourlyRate: 99,
    areas: ["Delhi", "Gurugram", "Noida", "Greater Noida", "Ghaziabad", "Faridabad"],
  };

  const T = {
    en: {
      nav_book: "Book", nav_partners: "Become a partner", nav_how: "How it works", nav_plans: "Plans",
      hero_eyebrow: "Delhi NCR · Event capture",
      hero_title: "Enjoy your own celebration. We'll capture it.",
      hero_lead: "Book a SnapAI capturer for your birthday, anniversary or family function. The SnapAI app captures the big moments automatically, your capturer adds the shots you ask for, and your private album is generated and sent to your WhatsApp instantly, the moment your event ends.",
      cta_book: "Book a capturer", cta_plans: "See plans",
      trust_1: "Starts at ₹99", trust_2: "Instant album on WhatsApp", trust_3: "Full refund if we don't show",
      badge_live: "AI auto-capture + manual shots",
      tagline: "Every event deserves one capturer.",
      how_eyebrow: "How it works", how_title: "Four steps, zero stress",
      s1_t: "Book online", s1_p: "Pick your date, place and plan in under a minute.",
      s2_t: "Capturer arrives", s2_p: "A verified SnapAI capturer reaches your venue on time.",
      s3_t: "AI + manual capture", s3_p: "The app auto-captures key moments; your capturer takes the special shots too.",
      s4_t: "Instant album", s4_p: "The moment your event ends, your capturer generates the album on the spot and it lands on your WhatsApp instantly, only for you and your family.",
      plans_eyebrow: "Plans", plans_title: "Simple, short, affordable",
      plans_sub: "Pay only for the time you need. Moment is the minimum standard we recommend for every event.",
      p30_name: "Moment", p60_name: "Highlight", p120_name: "Extended", p180_name: "Full Event",
      p_std: "Minimum standard", p_popular: "Most booked",
      p30_1: "Cake-cutting or one key ritual", p30_2: "AI picks the best shots", p30_3: "Instant album on WhatsApp",
      p60_1: "Entry, key moments and family photos", p60_2: "Candids of guests", p60_3: "Instant album on WhatsApp",
      p120_1: "Your whole small event", p120_2: "Group, family and candid shots", p120_3: "Instant album on WhatsApp",
      p180_1: "Multi-ritual or longer function", p180_2: "Extended AI + manual coverage", p180_3: "Instant album on WhatsApp",
      per_session: "/ session", choose: "Choose",
      p_custom_name: "Custom hours", p_custom_note: "Longer event? Pick your own hours — priced at ₹99/hour.",
      f_custom_hours: "Number of hours", custom_price_label: "Estimated price",
      promise_eyebrow: "Our promise", promise_title: "If your capturer doesn't show up, or the capture fails, you get a full refund.",
      promise_p: "We currently serve these areas. More cities soon.",
      gallery_eyebrow: "Moments we capture", gallery_title: "Birthdays, anniversaries, pujas and more",
      illustrative: "Illustrative images.",
      reviews_title: "Reviews", reviews_p: "Reviews coming soon. We're just starting in Delhi NCR, and every review here will be from a real customer.",
      faq_title: "Questions",
      q1: "What exactly does a SnapAI capturer do?", a1: "They come to your event with the SnapAI app. The app automatically captures important moments using AI, and the capturer also takes manual shots, like family photos or anything you ask for.",
      q2: "Do you do decoration or event planning?", a2: "No. SnapAI only captures your event. You enjoy it, we capture it.",
      q3: "How do I get my photos?", a3: "Instantly. The moment your event ends, your capturer generates the private album on the SnapAI app and sends it straight to your WhatsApp — no waiting. Only you can share it.",
      q4: "Which areas do you cover?", a4: "Delhi, Gurugram, Noida, Greater Noida, Ghaziabad and Faridabad.",
      q5: "What if the capturer doesn't come?", a5: "Full refund, no questions asked.",
      pcta_title: "Good with a camera? Earn with SnapAI.", pcta_btn: "Become a partner →",
      footer_about: "SnapAI is an AI-powered event capture service in Delhi NCR. Event capture only.",
      footer_links: "Links", footer_contact: "Contact", privacy: "Privacy Policy", terms: "Terms",
      wa_label: "WhatsApp us", wa_msg: "Hi SnapAI! I'd like to know more about booking a capturer.",
      wa_msg_partner: "Hi SnapAI! I'd like to join as a partner.",
      book_title: "Book a capturer", book_sub: "Takes about a minute. We'll call you to confirm.",
      step_event: "Your event", step_place: "Place & guests", step_plan: "Plan", step_you: "Your details",
      f_event_type: "Event type", f_date: "Event date", f_time: "Start time", f_area: "Area", f_locality: "Locality / venue area",
      f_guests: "Approx. number of guests", f_name: "Your name", f_phone: "Mobile number", f_email: "Email (optional)",
      f_wa: "Contact me on WhatsApp", f_consent: "I agree to receive a call/WhatsApp from SnapAI about my booking.",
      other_city: "Other city", ph_locality: "e.g. Dwarka Sector 12", ph_phone: "10-digit mobile",
      ev_birthday: "Birthday", ev_anniv: "Anniversary", ev_family: "Family function", ev_kitty: "Kitty party", ev_baby: "Baby shower", ev_corp: "Small corporate event", ev_other: "Other",
      outside_ncr: "We're not in your city yet. Submit anyway and we'll tell you first when we launch there.",
      back: "Back", next: "Next", submit_booking: "Confirm booking", sending: "Sending…",
      e_required: "Please fill this in.", e_phone: "Enter a valid 10-digit Indian mobile number.", e_consent: "Please tick the consent box.", e_generic: "Something went wrong. Please try again or WhatsApp us.",
      thanks_title: "You're booked in!", thanks_id: "Your booking ID", thanks_next: "Our team will call you within 15 minutes (9am–9pm) to confirm your capturer.",
      pay_now: "Pay now", talk_first: "Talk to us first", paid_title: "Payment received. Thank you!", waitlist_title: "You're on the waitlist",
      waitlist_p: "We'll let you know as soon as SnapAI launches in your city.", test_note: "Payments are in test mode during our pilot.",
      pt_eyebrow: "SnapAI Partners", pt_title: "Get booked to capture events near you.",
      pt_lead: "Photographers, event organisers, balloon decorators, students, or anyone with a good smartphone. Capture birthdays and family events with the SnapAI app and earn per event.",
      pt_join: "Register now", pt_who: "Who can join",
      w1: "Photographers (new or pro)", w2: "Event organisers", w3: "Balloon decorators", w4: "Students", w5: "Anyone with a good phone",
      pt_how: "How it works", ps1_t: "Register", ps1_p: "Fill the short form below.", ps2_t: "We call & verify", ps2_p: "A quick call to know you. Payout per event is shared on this call.",
      ps3_t: "Trial event", ps3_p: "Do one event with guidance from our team.", ps4_t: "Get bookings", ps4_p: "Receive bookings near you and capture with the SnapAI app.",
      pt_need: "What you need", need_1: "A smartphone with a good camera", need_2: "Reliability: reaching events on time", need_3: "Friendly manners with families",
      pf_title: "Partner registration", pf_business: "Business name (optional)", pf_type: "I am a", pf_exp: "Experience", pf_new: "New", pf_pro: "Pro",
      pf_areas: "Areas I can serve", pf_phone_model: "Smartphone model", pf_link: "Instagram / portfolio link (optional)",
      pf_consent: "I agree to receive a call/WhatsApp from SnapAI about partnering.", pf_submit: "Register as partner",
      t_photographer: "Photographer", t_organiser: "Event organiser", t_decorator: "Balloon decorator", t_student: "Student", t_other: "Other",
      pf_done_title: "Thanks for registering!", pf_done_p: "Your partner ID is {id}. We'll call you soon to verify and share the payout details.",
    },
    hi: {
      nav_book: "बुक करें", nav_partners: "पार्टनर बनें", nav_how: "कैसे काम करता है", nav_plans: "प्लान",
      hero_eyebrow: "दिल्ली NCR · इवेंट कैप्चर",
      hero_title: "आप अपना जश्न मनाइए। यादें हम कैद करेंगे।",
      hero_lead: "अपने बर्थडे, एनिवर्सरी या फैमिली फंक्शन के लिए SnapAI कैप्चरर बुक करें। SnapAI ऐप खास पलों को अपने आप कैप्चर करता है, कैप्चरर आपकी पसंद की फ़ोटो भी लेता है, और इवेंट खत्म होते ही आपका प्राइवेट एल्बम तुरंत आपके WhatsApp पर पहुंच जाता है।",
      cta_book: "कैप्चरर बुक करें", cta_plans: "प्लान देखें",
      trust_1: "सिर्फ़ ₹99 से शुरू", trust_2: "WhatsApp पर तुरंत एल्बम", trust_3: "न पहुंचे तो पूरा पैसा वापस",
      badge_live: "AI ऑटो-कैप्चर + मैनुअल फ़ोटो",
      tagline: "हर इवेंट में एक कैप्चरर ज़रूरी है।",
      how_eyebrow: "कैसे काम करता है", how_title: "चार आसान कदम",
      s1_t: "ऑनलाइन बुक करें", s1_p: "एक मिनट में तारीख, जगह और प्लान चुनें।",
      s2_t: "कैप्चरर पहुंचेगा", s2_p: "वेरिफ़ाइड SnapAI कैप्चरर समय पर आपकी जगह पहुंचेगा।",
      s3_t: "AI + मैनुअल कैप्चर", s3_p: "ऐप खास पल अपने आप कैप्चर करता है, कैप्चरर खास फ़ोटो भी लेता है।",
      s4_t: "तुरंत एल्बम", s4_p: "इवेंट खत्म होते ही कैप्चरर वहीं एल्बम बनाकर तुरंत आपके WhatsApp पर भेज देगा, सिर्फ़ आपके परिवार के लिए।",
      plans_eyebrow: "प्लान", plans_title: "आसान, छोटे और किफ़ायती",
      plans_sub: "जितना समय चाहिए, उतना ही भुगतान। हर इवेंट के लिए हम कम से कम 'मोमेंट' प्लान की सलाह देते हैं।",
      p30_name: "मोमेंट", p60_name: "हाइलाइट", p120_name: "एक्सटेंडेड", p180_name: "फ़ुल इवेंट",
      p_std: "न्यूनतम स्टैंडर्ड", p_popular: "सबसे ज़्यादा बुक",
      p30_1: "केक कटिंग या एक खास रस्म", p30_2: "AI सबसे अच्छी फ़ोटो चुनता है", p30_3: "WhatsApp पर तुरंत एल्बम",
      p60_1: "एंट्री, खास पल और फ़ैमिली फ़ोटो", p60_2: "मेहमानों की कैंडिड फ़ोटो", p60_3: "WhatsApp पर तुरंत एल्बम",
      p120_1: "आपका पूरा छोटा इवेंट", p120_2: "ग्रुप, फ़ैमिली और कैंडिड फ़ोटो", p120_3: "WhatsApp पर तुरंत एल्बम",
      p180_1: "कई रस्में या लंबा फ़ंक्शन", p180_2: "ज़्यादा AI + मैनुअल कवरेज", p180_3: "WhatsApp पर तुरंत एल्बम",
      per_session: "/ सेशन", choose: "चुनें",
      p_custom_name: "कस्टम घंटे", p_custom_note: "इवेंट लंबा है? खुद घंटे चुनें — ₹99/घंटा की दर से।",
      f_custom_hours: "कितने घंटे", custom_price_label: "अनुमानित कीमत",
      promise_eyebrow: "हमारा वादा", promise_title: "अगर कैप्चरर नहीं पहुंचा या कैप्चर फ़ेल हुआ, तो पूरा पैसा वापस।",
      promise_p: "अभी हम इन इलाकों में सेवा देते हैं। जल्द और शहरों में।",
      gallery_eyebrow: "जो पल हम कैद करते हैं", gallery_title: "बर्थडे, एनिवर्सरी, पूजा और भी बहुत कुछ",
      illustrative: "उदाहरण के लिए तस्वीरें।",
      reviews_title: "रिव्यू", reviews_p: "रिव्यू जल्द आ रहे हैं। हम अभी दिल्ली NCR में शुरू कर रहे हैं, और यहां हर रिव्यू असली ग्राहक का होगा।",
      faq_title: "सवाल-जवाब",
      q1: "SnapAI कैप्चरर असल में क्या करता है?", a1: "वह SnapAI ऐप के साथ आपके इवेंट में आता है। ऐप AI से खास पल अपने आप कैप्चर करता है, और कैप्चरर फ़ैमिली फ़ोटो या आपकी पसंद की फ़ोटो भी लेता है।",
      q2: "क्या आप डेकोरेशन या इवेंट प्लानिंग करते हैं?", a2: "नहीं। SnapAI सिर्f़सिर्f़ आपका इवेंट कैप्चर करता है। आप जश्न मनाइए, हम कैप्चर करेंगे।",
      q3: "फ़ोटो कैसे मिलेंगी?", a3: "तुरंत। इवेंट खत्म होते ही आपका कैप्चरर वहीं SnapAI ऐप में एल्f़बम बनाकर तुरंत आपके WhatsApp पर भेज देगा — कोई इंतज़़ार नहीं।",
      q4: "आप किन इलाकों में आते हैं?", a4: "दिल्f़ली, गुरुग्f़राम, नोएडा, ग्f़रेटर नोएडा, गाज़़ियाबाद और फ़रीदाबाद।",
      q5: "अगर कैप्f़चरर न आए तो?", a5: "पूरा पैसा वापस, बिना किसी सवाल के।",
      pcta_title: "फ़ोटो लेना अच्f़छा लगता है? SnapAI के साथ कमाएं।", pcta_btn: "पार्f़टनर बनें →",
      footer_about: "SnapAI दिल्f़ली NCR में AI-आधारित इवेंट कैप्f़चर सेवा है। सिर्f़फ़ इवेंट कैप्f़चर।",
      footer_links: "लिंक", footer_contact: "संप्f़क़", privacy: "प्f़राइवेसी पॉलिसी", terms: "नियम व शर्f़तें",
      wa_label: "WhatsApp करें", wa_msg: "नम्f़स्f़ते SnapAI! मुझे कैप्f़चरर बुक करने के बारे में जानना है।",
      wa_msg_partner: "नम्f़स्f़ते SnapAI! मैं पार्f़टनर बनना चाहता/चाहती हूं।",
      book_title: "कैप्f़चरर बुक करें", book_sub: "लगभग एक मिनट लगेगा। कन्f़फ़र्f़म करने के लिए हम आपको कॉल करेंगे।",
      step_event: "आपका इवेंट", step_place: "जगह और मेहमान", step_plan: "प्f़लान", step_you: "आपकी जानकारी",
      f_event_type: "इवेंट का प्f़रकार", f_date: "इवेंट की तारीख", f_time: "शुरू होने का समय", f_area: "इलाका", f_locality: "लोकैलिटी / वेन्f़यू",
      f_guests: "लगभग कितने मेहमान", f_name: "आपका नाम", f_phone: "मोबाइल नंबर", f_email: "ईमेल (ज़़रूरी नहीं)",
      f_wa: "मुझसे WhatsApp पर संप्f़क़ करें", f_consent: "मैं अपनी बुकिंग के बारे में SnapAI से कॉल/WhatsApp पाने के लिए सहमत हूं।",
      other_city: "कोई और शहर", ph_locality: "जैसे द्f़वारका सेक्f़टर 12", ph_phone: "10 अंकों का मोबाइल",
      ev_birthday: "बर्f़थडे", ev_anniv: "एनिवर्f़सरी", ev_family: "फ़ैमिली फ़ंक्f़शन", ev_kitty: "किट्f़ी पार्f़टी", ev_baby: "गोद भराई / बेबी शावर", ev_corp: "छोटा कॉर्f़पोरेट इवेंट", ev_other: "अन्f़य",
      outside_ncr: "हम अभी आपके शहर में नहीं हैं। फ़िर भी सब्f़मिट करें, वहां शुरू होते ही सबसे पहले आपको बताएंगे।",
      back: "पीछे", next: "आगे", submit_booking: "बुकिंग कन्f़फ़र्f़म करें", sending: "भेज रहे हैं…",
      e_required: "कृपया यह भरें।", e_phone: "सही 10 अंकों का भारतीय मोबाइल नंबर डालें।", e_consent: "कृपया सहमति वाला बॉक्f़स टिक करें।", e_generic: "कुछ गड्f़बज़़ हो गई। दोबारा कोशिश करें या हमें WhatsApp करें।",
      thanks_title: "आपकी बुकिंग हो गई!", thanks_id: "आपकी बुकिंग ID", thanks_next: "हमारी टीम 15 मिनट में (सुबह 9 से रात 9 बजे तक) कॉल करके आपका कैप्f़चरर कन्f़फ़र्f़म करेगी।",
      pay_now: "अभी भुगतान करें", talk_first: "पहले बात करें", paid_title: "भुगतान मिल गया। धन्f़यवाद!", waitlist_title: "आप वेटलिस्f़ट में हैं",
      waitlist_p: "आपके शहर में SnapAI शुरू होते ही हम आपको बताएंगे।", test_note: "पायलट के दौरान भुगतान टेस्f़ट मोड में हैं।",
      pt_eyebrow: "SnapAI पार्f़टनर", pt_title: "अपने आस-पास के इवेंट कैप्f़चर करें और कमाएं।",
      pt_lead: "फ़ोटोग्f़राफ़र, इवेंट ऑर्f़गानाइज़़र, बैलून डेकोरेटर, स्f़टूडेंट, या अच्f़छे स्f़मार्f़टफ़ोन वाला कोई भी। SnapAI ऐप से बर्f़थडे और फ़ैमिली इवेंट कैप्f़चर करें और हर इवेंट पर कमाएं।",
      pt_join: "अभी रजिस्f़टर करें", pt_who: "कौन जुड़ सकता है",
      w1: "फ़ोटोग्f़राफ़र (नए या प्f़रो)", w2: "इवेंट ऑर्f़गानाइज़़र", w3: "बैलून डेकोरेटर", w4: "स्f़टूडेंट", w5: "अच्f़छे फ़ोन वाला कोई भी",
      pt_how: "कैसे काम करता है", ps1_t: "रजिस्f़टर करें", ps1_p: "नीचे छोटा सा फ़ॉर्f़म भरें।", ps2_t: "कॉल और वेरिफ़िकेशन", ps2_p: "आपको जानने के लिए एक छोटी कॉल। हर इवेंट का पेमेंट इसी कॉल पर बताया जाएगा।",
      ps3_t: "ट्f़रायल इवेंट", ps3_p: "हमारी टीम की मदद से एक इवेंट करें।", ps4_t: "बुकिंग पाएं", ps4_p: "आस-पास की बुकिंग पाएं और SnapAI ऐप से कैप्f़चर करें।",
      pt_need: "आपको क्f़या चाहिए", need_1: "अच्f़छे कैमरे वाला स्f़मार्f़टफ़ोन", need_2: "भरोसेमंद होना: समय पर पहुंचना", need_3: "परिवारों के साथ अच्f़छा व्f़यवहार",
      pf_title: "पार्f़टनर रजिस्f़ट्f़रेशन", pf_business: "बिज़़नेस का नाम (ज़़रूरी नहीं)", pf_type: "मैं हूं", pf_exp: "अनुभव", pf_new: "नया", pf_pro: "प्f़रो",
      pf_areas: "जिन इलाकों में काम कर सकता/सकती हूं", pf_phone_model: "स्f़मार्f़टफ़ोन मॉडल", pf_link: "Instagram / पोर्f़टफ़ोलियो लिंक (ज़़रूरी नहीं)",
      pf_consent: "मैं पार्f़टनरशिप के बारे में SnapAI से कॉल/WhatsApp पाने के लिए सहमत हूं।", pf_submit: "पार्f़टनर के रूप में रजिस्f़टर करें",
      t_photographer: "फ़ोटोग्f़राफ़र", t_organiser: "इवेंट ऑर्f़गानाइज़़र", t_decorator: "बैलून डेकोरेटर", t_student: "स्f़टूडेंट", t_other: "अन्f़य",
      pf_done_title: "रजिस्f़टर करने के लिए धन्f़यवाद!", pf_done_p: "आपकी पार्f़टनर ID {id} है। वेरिफ़िकेशन और पेमेंट की जानकारी के लिए हम जल्f़द कॉल करेंगे।",
    },
  };

  function store(k, v) { try { if (v === undefined) return localStorage.getItem(k); localStorage.setItem(k, v); } catch (e) { return null; } }
  let lang = store("snap_lang") === "hi" ? "hi" : "en";
  const t = (k) => (T[lang] && T[lang][k]) || T.en[k] || k;

  function applyLang() {
    document.documentElement.lang = lang === "hi" ? "hi" : "en";
    document.querySelectorAll("[data-i18n]").forEach((el) => { el.textContent = t(el.dataset.i18n); });
    document.querySelectorAll("[data-i18n-ph]").forEach((el) => { el.placeholder = t(el.dataset.i18nPh); });
    document.querySelectorAll(".lang button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.lang === lang)));
    const wa = document.querySelector(".wa-float");
    if (wa) { wa.href = waLink(); wa.querySelector("span").textContent = t("wa_label"); }
    document.dispatchEvent(new CustomEvent("snap:lang", { detail: lang }));
  }
  function setLang(l) { lang = l; store("snap_lang", l); applyLang(); }

  const isPartnerPage = () => document.body.classList.contains("partners");
  function waLink(msgKey) {
    const msg = t(msgKey || (isPartnerPage() ? "wa_msg_partner" : "wa_msg"));
    return `https://wa.me/${CONFIG.whatsapp}?text=${encodeURIComponent(msg)}`;
  }

  function captureSource() {
    let saved = null;
    try { saved = JSON.parse(sessionStorage.getItem("snap_src") || "null"); } catch (e) {}
    if (saved) return saved;
    const q = new URLSearchParams(location.search);
    const utm = {};
    ["utm_source", "utm_medium", "utm_campaign", "utm_content", "fbclid", "ref"].forEach((k) => { if (q.get(k)) utm[k] = q.get(k); });
    const s = ((utm.utm_source || "") + " " + (document.referrer || "")).toLowerCase();
    let source = "Website";
    if (utm.fbclid || /facebook|fb|meta/.test(s)) source = "Meta Ad";
    else if (/instagram|ig/.test(s)) source = "Instagram";
    else if (/whatsapp|wa\b/.test(s)) source = "WhatsApp";
    else if (utm.ref || /referral/.test(s)) source = "Referral";
    const out = { source, utm };
    try { sessionStorage.setItem("snap_src", JSON.stringify(out)); } catch (e) {}
    return out;
  }

  async function rpc(fn, payload) {
    const res = await fetch(`${CONFIG.supabaseUrl}/rest/v1/rpc/${fn}`, {
      method: "POST",
      headers: { apikey: CONFIG.anonKey, Authorization: `Bearer ${CONFIG.anonKey}`, "Content-Type": "application/json" },
      body: JSON.stringify({ p: payload }),
    });
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  }
  function syncLead(type, id, event, token) {
    return fetch(`${CONFIG.supabaseUrl}/functions/v1/sync-lead`, {
      method: "POST",
      headers: { Authorization: `Bearer ${token || CONFIG.anonKey}`, apikey: CONFIG.anonKey, "Content-Type": "application/json" },
      body: JSON.stringify({ type, id, event }),
      keepalive: true,
    }).catch(() => {});
  }

  const imgUrl = (slot) => `${CONFIG.supabaseUrl}/storage/v1/object/public/site/${slot}.jpg`;
  function loadPhotos() {
    document.querySelectorAll("[data-photo]").forEach((box) => {
      const img = document.createElement("img");
      img.alt = box.dataset.alt || "";
      box.classList.add("empty");
      img.onload = () => box.classList.remove("empty");
      img.onerror = () => box.classList.add("empty");
      box.prepend(img);
      img.src = imgUrl(box.dataset.photo);
    });
  }

  const WA_ICON = '<svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M12 2a10 10 0 0 0-8.6 15.1L2 22l5-1.3A10 10 0 1 0 12 2Zm0 18.2a8.2 8.2 0 0 1-4.2-1.2l-.3-.2-3 .8.8-2.9-.2-.3A8.2 8.2 0 1 1 12 20.2Zm4.5-6.1c-.2-.1-1.5-.7-1.7-.8s-.4-.1-.6.1-.7.8-.8 1-.3.2-.5.1a6.7 6.7 0 0 1-3.3-2.9c-.3-.4.3-.4.7-1.3.1-.2 0-.3 0-.4l-.8-1.9c-.2-.5-.4-.4-.6-.4h-.5a1 1 0 0 0-.7.3 3 3 0 0 0-.9 2.2 5.2 5.2 0 0 0 1.1 2.8 11.9 11.9 0 0 0 4.6 4c1.7.7 2.4.8 3.2.6a2.8 2.8 0 0 0 1.8-1.3 2.3 2.3 0 0 0 .2-1.3c-.1-.1-.2-.2-.5-.3Z"/></svg>';

  function chrome() {
    const header = document.getElementById("site-header");
    if (header) header.innerHTML = `
      <div class="wrap">
        <a class="brand" href="/"><img src="/assets/logo-wordmark.png" alt="SnapAI" class="brand-logo"></a>
        <nav class="nav" aria-label="Main">
          <a class="hide-sm" href="/#how" data-i18n="nav_how"></a>
          <a class="hide-sm" href="/#plans" data-i18n="nav_plans"></a>
          <a class="hide-sm" href="/partners" data-i18n="nav_partners"></a>
          <div class="lang" role="group" aria-label="Language">
            <button type="button" data-lang="en">EN</button><button type="button" data-lang="hi">हिं</button>
          </div>
          <a class="btn btn-primary btn-sm" href="/book" data-i18n="nav_book"></a>
        </nav>
      </div>`;
    const footer = document.getElementById("site-footer");
    if (footer) footer.innerHTML = `
      <div class="wrap cols">
        <div><div class="brand"><img src="/assets/logo-wordmark.png" alt="SnapAI" class="brand-logo"></div>
          <p style="margin-top:10px" data-i18n="footer_about"></p><p><em data-i18n="tagline"></em></p></div>
        <div><h3 data-i18n="footer_links"></h3>
          <p><a href="/book" data-i18n="nav_book"></a><br><a href="/partners" data-i18n="nav_partners"></a><br>
          <a href="/privacy" data-i18n="privacy"></a><br><a href="/terms" data-i18n="terms"></a></p></div>
        <div><h3 data-i18n="footer_contact"></h3>
          <p>WhatsApp: <a href="https://wa.me/${CONFIG.whatsapp}">${CONFIG.phoneDisplay}</a><br>
          <a href="${CONFIG.instagram}" rel="noopener" target="_blank">Instagram</a> · <a href="${CONFIG.messenger}" rel="noopener" target="_blank">Messenger</a>
          ${CONFIG.email ? `<br><a href="mailto:${CONFIG.email}">${CONFIG.email}</a>` : ""}</p>
          <p style="font-size:.85rem">© ${new Date().getFullYear()} SnapAI · Delhi NCR</p></div>
      </div>`;
    if (!document.querySelector(".wa-float") && !document.body.dataset.noWa) {
      const a = document.createElement("a");
      a.className = "wa-float"; a.target = "_blank"; a.rel = "noopener";
      a.innerHTML = WA_ICON + "<span></span>";
      document.body.appendChild(a);
    }
    document.querySelectorAll(".lang button").forEach((b) => b.addEventListener("click", () => setLang(b.dataset.lang)));
  }

  function init() { chrome(); captureSource(); loadPhotos(); applyLang(); }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();

  return { CONFIG, t, get lang() { return lang; }, setLang, rpc, syncLead, captureSource, waLink, imgUrl };
})();
